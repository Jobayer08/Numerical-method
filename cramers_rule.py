import numpy as np
A=np.array([[1,2,3],[0,1,4],[5,6,0]])
B=np.array([9,8,7])
D=np.linalg.det(A)
if D==0:
    print("The system has no unique solution.")
else:
    Dx=np.linalg.det(np.column_stack((B,A[:,1],A[:,2])))
    Dy=np.linalg.det(np.column_stack((A[:,0],B,A[:,2])))
    Dz=np.linalg.det(np.column_stack((A[:,0],A[:,1],B)))
x=Dx/D
y=Dy/D
z=Dz/D
print(f"The solution is x={x:.3f}, y={y:.3f}, z={z:.3f}")    