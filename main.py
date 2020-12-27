import random as rd
import matplotlib.pylab as plt
from numpy.random import default_rng
from scipy.sparse import dok_matrix
import numpy as np
import time
import argparse

rng = default_rng()
def simulate(n, n_steps, d_step, a_min, a_max):
    """
    default a semester of 5 minutes steps for 286 rares mobs
    """
    day = 0
    hour = 0
    minute = 0
    next_respawn = {}
    alive = list(range(n)) # everybody alive at the 1st hour
    long_alive = []
    future_image = dok_matrix((n_steps,n))
    pauses = []
    cemetery = []
    spawned_during_maj = {}
    paused=False
    alive_count = []
    p_id = 0

    t = time.time()
    print("Simulating ...", "#", n_steps)
    
    DEAD_DURING_MAINTENANCE = -0.5
    ALIVE = 1
    ALIVE_DURING_MAINTENANCE = 2
    ALL_DEAD = -2
    
    for i in range(n_steps):
        if i%10000 == 0:
            print("#",i)
        paused =  day == 1 and hour > 8 and hour < 12 # TODO: estimate a duration of the maintenance
        was_paused =  day == 1 and hour - 1 > 8 and hour - 1 < 12
        if paused:
            pauses.append(i)
            for j in range(n):
                future_image[i,j] = DEAD_DURING_MAINTENANCE
            if not was_paused:
                p_id += 1
        for a in alive:
            long_alive.append(a)
            if paused:
                future_image[i,a] = ALIVE_DURING_MAINTENANCE
                if not a in spawned_during_maj:
                    spawned_during_maj[a] = 0
                spawned_during_maj[a] += 1
            else:
                future_image[i,a] = ALIVE
        alive = []
        for a in next_respawn:
            next_respawn[a] -= 1
            if next_respawn[a] == 0:
                alive.append(a)
        if len(alive) == 0 and len(long_alive) == 0:
            for p in range(n):
                cemetery.append(i)
                future_image[i,p] = ALL_DEAD
        minute += d_step
        minute %= 60
        if minute == 0:
            hour += 1
            hour %= 24
        if hour == 0:    
            day += 1
            day %= 7

        if not paused:
            for a in long_alive:
                next_respawn[a] = rd.randint(a_min*60/d_step,a_max*60/d_step)
            long_alive = []
        else:
            for a in long_alive:
                future_image[i,a] = 1.5
    print("Simulation finished !", time.time() - t, "s")

    def mean(l):
        s = 0
        for i in l:
            s += i
        return s / len(l)

    def var(l):
        return mean([i**2 for i in l]) - mean(l)**2

    t = time.time()
    print("Building the distribution...")
    distribution = np.zeros((n_steps-1))
    distribution_maj = np.zeros((n_steps-1))
    day = 1 
    hour = 0
    minute = d_step
    # we account for sparsity : good
    s = np.sum(future_image > 0, axis=1) 
    for i in range(1,n_steps):
        if i%10000==0:
            print("#",i,"(",time.time() - t, ")")
        paused =  day == 1 and hour > 8 and hour < 12
        just_paused = day == 1 and hour == 11
        if not paused:
            distribution[i-1] += s[i]  
        if just_paused:
            distribution_maj[i-1] += s[i]
        minute += d_step
        minute %= 60
        if minute == 0:
            hour += 1
            hour %= 24
        if hour == 0:
            day += 1
            day %= 7
    print("Distributions built!", time.time() - t, "s")
    return future_image, distribution, distribution_maj

def lst_wo(l, v):
    return [i for i in l if i != v]

def plot_dist(dist):
    plt.figure()
    plt.hist(dist, bins=30, range=(0,30))
    plt.title("# tranches de 5 minutes dans lesquelles il y a x archis en jeu")
    plt.show(block=False)

def plot_maj(dist_maj):
    plt.figure()
    plt.hist(lst_wo(dist_maj, 0))
    plt.title("# tranches avec x archis en jeu en sortie de maintenance")
    plt.show(block=False)

def plot_days(dist,n_steps,d_step):
    steps_per_day = 24*60/d_step
    tst = np.floor( np.arange(0, n_steps-1, 1) * (1/steps_per_day) )
    distrib_days = []
    for j in range(7):
        distrib_days.append(dist[tst == j,])
    mx = -1
    for j in range(7):
        unique, counts= np.unique(distrib_days[j], return_counts=True)
        dct = dict(zip(unique,counts))
        for k,m in dct.items():
            if mx < m:
                mx = m
    plt.figure()
    ax = plt.subplot(331)
    ax.hist(distrib_days[0], bins=30, range=(0,30))
    ax.set_title("Jour #0 de la semaine")
    plt.ylim(0,mx*1.1)
    for j in range(1,7):
        a_ = plt.subplot(331+j)
        a_.hist(distrib_days[j], bins=30, range=(0,30))
        a_.set_title("Jour #"+str(j)+" de la semaine")
        plt.ylim(0,mx*1.1)
    plt.ylim(0,mx*1.1)
    plt.show(block=False)

def plot_simu(img):
    plt.figure()
    plt.imshow(img.todense(),interpolation="none",aspect="equal",cmap="BuPu")
    plt.title("1 colonne = 1 archi, 1 ligne = 5 minutes")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description='Simulation archimonstres Dofus')
    parser.add_argument("-a", "--monsters", action="store", help="Nombres de monstres", default=286, type=int)
    parser.add_argument("-m", "--min", action="store", help="Délai de repop mini (heure)", default=6, type=int)
    parser.add_argument("-M", "--max", action="store", help="Délai de repop maxi (heure)", default=18, type=int)
    parser.add_argument("-n", "--steps", action="store", help="Nombre d'étapes", default=int(6/12*365*24*60/5), type=int)
    parser.add_argument("-d", "--d_step", action="store", help="Pas de temps (minutes)", default=5, type=int)
    args = parser.parse_args()
    print("args:",args)
    img, dist, dist_maj = simulate(args.monsters, args.steps, args.d_step, args.min, args.max)
    
    print("ocre générables:",np.min(np.sum(img > 0,axis=0)[0]))
    
    t = time.time()
    plot_dist(dist)
    print("dist:", time.time() - t, "s")

    t = time.time()
    plot_maj(dist_maj)
    print("dist_maj:", time.time() - t, "s")
    
    t = time.time()
    plot_days(dist, args.steps, args.d_step)
    print("dist_day:", time.time() - t, "s")
    
    t = time.time()
    plot_simu(img)
    print("dense img:", time.time() - t, "s")
