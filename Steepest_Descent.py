import numpy as np
import time

def quadraticFunction_f(A: np.array,b: np.array,c: np.float64,X: np.array) -> np.float64:
    """
    Quadratic function with variable X. 
    A is (n,n) b is (n,1) X is (n,1) and c is a scalar 
    Computes \sum_{i,j} a_{i,j} x_i x_j  - \sum_{i} x_i b_i + c 
    """
    return np.dot(X,np.dot(A,X))- np.dot(b,X)+c

def minusGradient_f(A: np.array,b: np.array,X: np.array) -> np.float64:
    """
    Represents a linear function with matrix A of (n,n) and b vector o
    """
    return b-np.dot(A,X)


def steepest_descent(A,b,c,x0,epsilon):
    ri=minusGradient_f(A,b,x0)
    xi=x0
    i=0
    while np.linalg.norm(ri)>=epsilon: # If the norm of the gradient is less than epsilon, we stop
        print("gradient",ri)
        print("current point",xi)

        print("Function value", quadraticFunction_f(A,b,c,xi))
        print()
        Ar=np.dot(A,ri)
        rr=np.dot(ri,ri)
        rAr=np.dot(ri,Ar)
        alphai= rr/rAr
        xi=xi+alphai*ri
        ri=ri-alphai*Ar
        i=i+1
        print(i)


A=np.array([[3,2],[2,6]])
b=np.array([2,-8])
c=0
x0=[0,0]
epsilon=0.00001
steepest_descent(A,b,c,x0,epsilon)


def matrix_notation_polynom_squareloss(x,y,d):
    data_size=len(x)
    A=np.zeros((d,d))
    for i  in range(d):
        for j in range(d):
            A[i][j]=(sum(x**(i+j)))
    A=A/data_size
    b=np.zeros(d)
    for i in range(d):
        b[i]=sum((x**i)*y)
    b=2*b/data_size
    c=sum(y**2)/data_size
    return  (A,b,c)


def objective_function(A,b,c,w):
    return  (np.dot(w,np.dot(A,w)) - np.dot(b,w) +c)



def full_gradient(A,b,w):
    return 2*np.dot(A,w)-b

def conjugateGradient(A,b,x,epsilon):
    d = b - np.dot(A,x)
    r = d
    i = 0
#     while(np.linalg.norm(r) > epsilon):
    while i < epsilon:
        alpha = np.dot(r,r)/np.dot(d,np.dot(A,d)) # alpha i
        x = x + np.dot(alpha , d) # x(i + 1)
        r_next = r - np.dot(alpha, np.dot(A , d))
        beta = np.dot(r_next,r_next)/np.dot(r,r)
        d = r_next + np.dot(beta,d)
        r = r_next
        i = i + 1
        
#     print("CG steps: ",i)
    return x


def newton_method(x,y,d,w0,alpha,epsilon,k):
    u=matrix_notation_polynom_squareloss(x,y,d)
    A=u[0]
    b=u[1]
    c=u[2]
    w=w0
    gr=full_gradient(A,b,w)
    i=0
    while objective_function(A,b,c,w) >= epsilon and i < 10000000:
        x0=np.zeros(d)
        gr=full_gradient(A,b,w)
        sk=conjugateGradient(A,-gr,x0,k)
        w=w + alpha*sk
        i=i+1
    return i,w,objective_function(A,b,c,w)


x=np.arange(1,11)

y1 = [1 + 9*x + 10*x**2,3]
y2 = [2 + 5*x + 8*x**2,3]
y3 = [3 + 6*x + 2*x**2,3]

y4 = [5 + 9*x + 4*x**2 + 6*x**3 + 3*x**4,5]
y5 = [1 + 1*x + 2*x**2 + 3*x**3 + 4*x**4,5]
y6 = [2 + 6*x + 9*x**2 + 5*x**3 + 1*x**4,5]

y7 = [1 + 6*x + 8*x**2 + 2*x**3 + 3*x**4 + 10*x**5 + 2*x**6,7]
y8 = [2 + 10*x + 8*x**2 + 1*x**3 + 7*x**4 + 3*x**5 + 9*x**6,7]
y9 = [5 + 9*x + 10*x**2 + 4*x**3 + 5*x**4 + 5*x**5 + 1*x**6,7]

ys = [y1,y2,y3,y4,y5,y6,y7,y8,y9]
# ys =[y7,y8,y9]


i = 7
for y in ys:
    d = y[1]
    for k in range(1,d+1):
        print("==============")
        start = time.time()
        iterations, final_w, final_cost = newton_method(x,y[0],d,np.zeros(d),0.01,0.1,k)
        end = time.time()
        print("Function: y",i)
        print("K: ",k)
        print("Iterations: ", iterations)
        print("Final W: ",final_w)
        print("Final Cost: ", final_cost)
        print("Time: ",end - start)
        
        report = np.array([i,k,iterations,final_w,final_cost,end-start])
        np.save('outputs/i_'+str(i)+'_k_' + str(k) + '.out', report)
    i = i + 1
#         print()

