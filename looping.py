import numpy as np #NumPy est une bibliothèque pour langage de programmation Python, destinée à manipuler des matrices ou tableaux multidimensionnels ainsi que des fonctions mathématiques opérant sur ces tableaux.
from scipy.integrate import odeint # Scipy.integrate permet de resoudre des equa diff
import matplotlib.pyplot as plt #Matplotlib est une bibliothèque du langage de programmation Python destinée à tracer et visualiser des données sous formes de graphiques.
from matplotlib import cm 


#vitesse au bas de la pente sans frottements
def FVbaspente(g,Hpente):
    return np.sqrt(2*g*Hpente)

#vitesse au bas de la pente avec frottements
def FVbaspenteAvecFrottements(x,g,Ud): 
    x=1.44 #la distance parcourue sur la pente
    return np.sqrt(2*x * g * (np.sin(40) - Ud*np.cos(40)))

def FVbaspenteAvecFrottementsH(x,g,Ud,alpha): 
    x=1.44 #la distance parcourue sur la pente
    alpha=40 #l'angle de la pente
    return np.sqrt(2*x * g * (np.sin(alpha) - Ud*np.cos(alpha)))
# Ici nous avons les valeurs initiales
#La résolution des équations du mouvement de la voiture dans le looping avec frottements à l'aide d'une méthode numérique

# Calcul de la vitesse de la voiture dans le looping en prenant en compte les frottements

def Vloop(vloop, t):
    
    eqdiff = [vloop[1] , (1/(m*R)*(-m*g*(np.sin(vloop[0])+Ud*np.cos(vloop[0]))-(vloop[1])**2*(Ud*m*R+(pair/2)*S1*Cx*R**2)))]
    
    return eqdiff
#vitesse minimale pour passer le looping sans tomber 
def FVinitloop(g,R):
    return np.sqrt(4*g*R)

#hauteur minimale pour passer le looping sans tomber 
def Hminiloop(g):
    return (FVinitloop(g,R))**2/(2*g)

#Calcul de la vitesse initiale pour passer le ravin
def FVinitravin(Hravin,Dravin,g):
    return np.sqrt((-g*(Dravin)**2)/(-2*Hravin))
def trajravin(frtravin, tr):
    
    eqdiff = [frtravin[2],frtravin[3],-(pair/(2*m))*np.sqrt(frtravin[2]**2+frtravin[3]**2)*S1Cx*frtravin[2]+(pair/(2*m))*np.sqrt(frtravin[2]**2+frtravin[3]**2)*S2Cz*frtravin[3],-(pair/(2*m))*np.sqrt(frtravin[2]**2+frtravin[3]**2)*S1Cx*frtravin[3]+(pair/(2*m))*np.sqrt(frtravin[2]**2+frtravin[3]**2)*S2Cz*frtravin[2]-g]
    
    return eqdiff



#variables
Ud=0.01 #frotements du sol
m=1760 #masse voiture (kg)
Hpente = 20 #hauteur de la pente (m)
alpha=(40*np.pi)/180 #angle de la pente (radiant)
Hloop=0.23 #hauteur du looping (m)
R=0.115 #rayon du looping (m)
Hravin=1 #hauteur du ravin (m)
Dravin=9 #longueur du ravin (m)
g=9.81 #constante gravitationelle (m.s*10**-2)
pair = 1.225   # masse volumique de l'air (kg.m^-3)
Cx=0.04 #frottement de l'air 
S1=1.95*1.35 #surface de la voiture 
V0 = 4.5                # Vitesse initiale 
vloop = [0, V0/R] 
t =  np.linspace(0,1,100)
# dans le ravin:
S1Cx = 0.001
S2Cz = 0.01
V = 4.3
frtravin0 = [0,0,V,0]
frtravin = odeint(trajravin, frtravin0, t)

print("vitesse bas de la pente: ", FVbaspente(g,Hpente))

x = np.linspace(0,Hpente,20)
y = FVbaspente(g,x)


plt.plot(x,y)
plt.title("vitesse en fonction de la hauteur sans frottement de la pente")
plt.xlabel("vitesse")
plt.ylabel("hauteur")
plt.grid()

plt.show()

print("vitesse bas de la pente avec frottement: ", FVbaspenteAvecFrottements(x,g,Ud))
print("vitesse bas de la pente avec frottement: ", FVbaspenteAvecFrottementsH(x,g,Ud,alpha))

"""
tracé de la vitesse en fonction de la hauteur sans frottement pente
"""
x = np.linspace(0,alpha,20)#remplace par ce que t'apelle hauteur
y = FVbaspente(g,x)


plt.plot(x,y)
plt.title("vitesse en fonction de la hauteur avec frottement de la pente")
plt.xlabel("vitesse")
plt.ylabel("hauteur (angle)")
plt.grid()

plt.show()
# affichage du graphique

vloop = odeint(Vloop,vloop,t)


plt.plot(vloop[:,0],vloop[:,1]*R, color = "blue")
plt.title("Vitesse de la voiture dans le looping")
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse en m/s")
plt.grid()
plt.show()

print("vitesse initiale du looping sans frottements: ", FVinitloop(g,R))
print("hauteur initiale de la pente pour passer le looping: ",Hminiloop(g))
print("vitesse finale/de sortie du looping sans frottements: ", FVinitloop(g,R))
print("vitesse initiale ravin:  ", FVinitravin(Hravin,Dravin,g))

# Calcul pour la partie du ravin
#La résolution des équations du mouvement de la voiture lâchée dans le ravin avec une vitesse initiale sans frottements puis avec frottements 

travin = np.linspace(0,0.5,100)
V0 = FVinitravin(Hravin,Dravin,g)      #vitesse initiale


# Le tracé de la trajectoire de la voiture dans le ravin sans frottements

xravin = V0*travin

yravin = -0.5*g*travin**2


# affichage du graphique de la trajectoire de la voiture dans le ravin sans  frottements

plt.plot(xravin, yravin, color = "black")
plt.title("Trajectoire dela voiture dans le ravin")
plt.xlabel("x(m)")
plt.ylabel("y(m)")

plt.show()


plt.plot(frtravin[:,0], frtravin[:,1], color = "black")
plt.title("Trajectoire dela voiture dans le ravin")
plt.xlabel("x(m)")
plt.ylabel("y(m)")

plt.show()

#bilan graphique de la trajectoire de la voiture dans le ravin sans et avec frottements

plt.plot(xravin, yravin, color = "green")
plt.plot(frtravin[:,0], frtravin[:,1], color = "red")
plt.title("Trajectoire dela voiture dans le ravin")
plt.xlabel("x(m)")
plt.ylabel("y(m)")

plt.show()

print("Sans prendre en compte les frottements en vert, en prenant en compte les frottements en rouge")
