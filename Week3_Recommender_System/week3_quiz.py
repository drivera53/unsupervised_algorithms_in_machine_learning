import numpy as np
# Compute the Jaccard distance between mike0702 and dadvador
ratings = np.array([[3, 3, 4, 0, 4, 2, 3, 0],
                    [3, 5, 4, 3, 3, 0, 0, 4],
                    [0, 4, 0, 5, 0, 0, 2, 1],
                    [2, 0, 0, 4, 0, 4, 4, 5]])
Ub = (ratings>0).astype(int)
print(Ub)

def jaccard(a,b):
    return (a*b).sum()/((a+b)>0).sum()

users = ["firechicken","mike0702","zephyros","dadvador"]

simmat=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        simmat[i,j] = jaccard(Ub[i],Ub[j])
        if i<j:
            print(users[i]+'-'+users[j], jaccard(Ub[i],Ub[j]))

print(simmat)

# Compute the Cosine  distance between firechicken and mike0702
def cos(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

simmat=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        simmat[i,j] = cos(Ub[i],Ub[j])
        if i<j:
            print(users[i]+'-'+users[j], cos(Ub[i],Ub[j]))
print(simmat)

Ur = (ratings>=3).astype(int)
print(Ur)

# Now, let's convert user ratings to good vs. bad. We consider it's good (1) if the rating is 3 or above, and bad (0) otherwise. Compute the Jaccard distance between zephyros and dadvador.
simmat=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        simmat[i,j] = jaccard(Ur[i],Ur[j])
        if i<j:
            print(users[i]+'-'+users[j], jaccard(Ur[i],Ur[j]))