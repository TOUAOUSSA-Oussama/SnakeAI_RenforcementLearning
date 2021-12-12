import torch
from torch._C import set_anomaly_enabled
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# A feedforward net with one input layer, one hidden layer, one output layer
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    
    def save(self, file_name="model.pth"):
        model_folder_path = './model' 
        # creer un dossier model s'il n'est pas deja creer
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # enregistrer le modele dans ce dossier :
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Train the model     
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        # compile the model :
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        # Convertir les parametres en des tenseurs 
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        
        n = len(state.shape) # nombre d'echantillon qu'on a 
        
        # dans le cas de train_short_memory : chaque element est de la forme (x)
        # dans le cas de train_long_memory : chaque element est de la forme (n, x)
        if n == 1 :
            # rendre la forme des elements (1=taille du batch, x=valeur de l'elmt)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )
        
        # 0 : on calcule la prediction du modele de l'etat actuel :
        pred = self.model(state) # c'est une image de Q = qualite de l'action de l'etat actuel
        target = pred.clone() # c'est pour stocker ensuite Q_new 
        
        # pour chaque echantillon :
        for idx in range(len(done)):
            # 1 : on predit Q (qualite de l'action=reward) liee a l'etat actuel :
            Q_new = reward[idx] 
            # on fait la mise a jour sauf si le jeu n'est pas termine (game_over = False )
            if not done[idx]:
                # 2 : mise a jour de Q : Q_new = Q + gamma * max(model.predict(nouvelle etat))
                Q_new =  Q_new + self.gamma*torch.max(self.model(next_state[idx]))
                
            # metter a jour la case correspond a ce Q dans target :
            target[idx][torch.argmax(action).item()] = Q_new
        
        # optimisation :
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
            
          