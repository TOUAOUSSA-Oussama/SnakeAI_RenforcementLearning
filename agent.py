import torch
import random
import numpy as np
from collections import deque # permet de stocker les donnees (reward, game_over, ...)
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Fixation de quelque parametres :
MAX_MEMORY = 100_100 # taille max de deque
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # pour random
        self.gamma = 0.9 # = discount rate pour la mise a jour de Q qui doit etre entre 0 et 1
        self.memory = deque(maxlen=MAX_MEMORY) # lorsque la memoire depasse MAX_MEMORY, deque appelle automatiquement popleft() qui supprime le premier element (le plus ancien) de la deque
        self.model = Linear_QNet(11, 256, 3) # 11 = taille de state, 256 = taille de hidden layer, 3 = taille de l'action
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    # Calculer l'etat de l'agent dans game :
    def get_state(self, game):
        # ici on calcule les 11 valeurs de 0 ou 1 :
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    # Memoriser l'etat et l'action et reward et la prochaine etat et game_over de l'agent
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # si memory est plein (taille > MAX_MEMORY), le premier ellement sera automatiquement supprimer (popleft)
        # par consequent, self.memory = (elemnt1, element2, ....) avec elementi = (state_i, action_i, ...)


    # Entrainer le modele :
    # train for all game steps (des le depart jusqu'a game over)
    def train_long_memory(self):
        # tout d'abord, diviser l'ensemble des elements de self.memory en des batchs de taille Batch_size
        if len(self.memory) > BATCH_SIZE :
            mini_sample = random.sample(self.memory, BATCH_SIZE) # chaque batch est de taille 1000 element avec element = (state, action, ....)
        else : 
            # ne rien faire si le nombre des elements est inferieur a la taille du batch
            mini_sample = self.memory

        # rassembler les states, actions, ... pour avoir en sortie chaque batch = [1000 state, 1000 action,....]
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Faire l'entrainement :
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    # train for one game step
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Calculer l'action que va faire l'agent a partir d'une etat :
    def get_action(self, state):
        # random moves : tradeoff exploration => sauf au debut tant le modele n'est pas encore bien entrainer
        self.epsilon = 80 - self.n_games# plus on avance c'est a dire le nombre des jeus augmente plus epsilon devient petit
        final_move = [0,0,0]# initialisation
        if random.randint(0, 200) < self.epsilon:
            # plus epsilon est petit (c'est a dire n_games est grand), 
            # plus la proba d'avoir un nombre < a epsilon est petite et 
            # donc devient moins probable de faire un random move 
            # si n_games depasse 80, alors epsilon sera negtive et alors on est sur de ne pas
            # faire un random move 
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # on predit l'action par le modele :
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)# ceci donne un vecteur de proba de taille 3
            move = torch.argmax(prediction).item()# on extrait l'indice de la case de proba plus grande
            final_move[move] = 1

        return final_move
                 

def train():
    # Initialisation :
    plot_scores = [] # pour tracer les scores
    plot_mean_scores = [] # moyenne des scores
    total_score = 0 # somme des scores 
    record = 0 # meilleur score
    agent = Agent()
    game = SnakeGameAI()
    # Boucle d'entrainement :
    while True :
        # get the old state of the agent :
        state_old = agent.get_state(game)

        # get the action (move) depending on state_old
        final_move = agent.get_action(state_old)     

        # perform the action (calcul des performances de cette action) puis get the new state of the agent :
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory = c'est a dire la memoire lier a une seul etape (etape qu'on vient de faire)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # memoriser les informations :
        agent.remember(state_old, final_move, reward, state_new, done)

        # dans le cas de game_over :
        if done :
            # Reinitialiser l'envirenement :
            game.reset()
    
            # train the long memory = c'est a dire la memoire lier a la partie de jeu des le debut (depart) jusqu'a la fin (game over)
            agent.train_long_memory()

            # Incrementer le nombre des jeus :
            agent.n_games += 1

            # Verifier si on a obtenu a meilleur score :
            if score > record :
                record = score
                agent.model.save() # enregistrer le modele obtenu

            # Afficher les informations liees a cette partie de jeu :
            print('-Game: ', agent.n_games, '-Score: ', score, '-Record: ', record)

            # Tracer les courbes pour visualiser l'avancement de l'entrainement :
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()