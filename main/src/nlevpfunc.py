"""
@author: aaborghi
This library includes some of the most famous the contour integral eigensolvers. 
"""
from numpy import zeros, concatenate, block, identity, matrix, diag, pi, transpose, linspace, mean, max
from numpy.linalg import eig, svd, inv, norm, cond
from scipy.linalg import solve

class contour_solver:
    '''
    n : number of dimensions of the eigenvalue problem
    
    T : nlevp
    
    p : parameter (still to implement)
    '''
    def __init__(self, n, T):
        self.n = n
        self.T = T

    
    def hankel(self, N, K, trunc, col, phi, dphi, V):
        '''
        N : number of quadrature nodes
        
        K : number of Ak
        
        trunc : truncation of the B matrix
        
        col : number of columns of the Hankel matrix
        
        phi, dphi: contour descriptor for the quadrature and its derivative
        
        V : random matrix
        
        Returns
        -------
        Lamb : eigenvalues

        circle: circle in which the contour integration is done
        
        eigvec: normalized eigenvectors
        '''
        
        n = self.n
        T = self.T
        A = zeros((n,1), dtype='complex_')
        
        for k in range(0, K):
            Ak = identity(n, dtype='complex_') * 0
            for i in range(0, N):
                Ak += solve(T(phi((2 * i * pi) / N)), (1 / (1j*N)) * V * (phi((2 * pi * i) / N) ** k) * dphi((2 * pi * i)/N))
            A = concatenate( (A,Ak) , axis=1)
            
        A = A[:,1:(n*K)+1] #list of all the Ak matrices

        Hank0 = zeros((1,n*col))
        Hank1 = zeros((1,n*col))

        for col_counter in range(0,col):
            Hank0 = block([[Hank0],[A[:,(col_counter * n):((col_counter+col) * n)]]])
            Hank1 = block([[Hank1],[A[:,((col_counter+1) * n):((col_counter+1+col) * n)]]])
            
        Hank0 = Hank0[1:(n*(col+1)),:]
        Hank1 = Hank1[1:(n*(col+1)),:]
        
        if trunc == 0:
            u, s, vh = svd(Hank0, full_matrices=True)
            sigma = diag(s)
            u = matrix(u)
            vh = matrix(vh)
        else:
            u, s, vh = svd(Hank0, full_matrices=True)
            sigma = diag(s[0:trunc])
            u = matrix(u[:,0:trunc])
            vh = matrix(vh[0:trunc,:])   
        
        B = u.H @ Hank1 @ vh.H @ inv(sigma)
        Lamb, v1 = eig(B)
        
        eigvec = u[0:n,:] @ v1
        
        x = linspace(0, 2*pi, N)
        circle = phi(x)
        
        return Lamb, circle, eigvec/norm(eigvec)
    
    
    def single_loewner(self, N, K, trunc, col, phi, dphi, sigma):
        '''
        N : number of quadrature nodes
    
        K : number of Ak
    
        trunc : truncation of the B matrix
    
        col : number of columns of the Hankel matrix
        
        phi : function describing the contour's border
        
        dphi : derivative of phi
    
        sigma : interpolation point
    
        Returns
        -------
        Lamb : eigenvalues
        
        circle: circle in which the contour integration is done
        
        eigvec: normalized eigenvectors
        '''
        n = self.n
        T = self.T
        M = zeros((n,1), dtype='complex_')
        circle = zeros((1,N), dtype='complex_')

        for k in range(0, K):
            Mk = identity(n, dtype = 'complex_') * 0
            for i in range(0, N):
                Mk += solve(T(phi((2 * i * pi) / N)), identity(n, dtype = 'complex_') * (1 / (1j*N)) * (((-1) ** k) / ((sigma - phi((2 * pi * i) / N)) ** (k+1))) * dphi((2 * pi * i)/N))
                circle[0,i] = phi((2 * i * pi) / N)
            M = concatenate( (M,Mk) , axis=1)
            
            
        M = M[:,1:(n*K)+1] #list of all the Mk matrices

        L0 = zeros((1,n*col))
        L = zeros((1,n*col))

        for col_counter in range(0,col):
            L0 = block([[L0],[M[:,(col_counter * n):((col_counter+col) * n)]]])
            L = block([[L],[M[:,((col_counter+1) * n):((col_counter+1+col) * n)]]])
        lenL0 = len(L0)
        L0 = L0[1:lenL0,:]
        lenL = len(L)
        L = L[1:lenL,:]
        Ls = (sigma * L) + L0
        
        if trunc == 0:
            u, s, vh = svd(L, full_matrices=True)
            sigma = diag(s)
            u = matrix(u)
            vh = matrix(vh)
        else:
            u, s, vh = svd(L, full_matrices=True)
            sigma = diag(s[0:trunc])
            u = matrix(u[:,0:trunc])
            vh = matrix(vh[0:trunc,:])   
        
        B = inv(sigma) @ u.H @ Ls @ vh.H 
        Lamb, Vright = eig(B)
        
        Cr = L0[0:n,:]
        eigvec = Cr @ vh.H @ Vright
        
        return Lamb, circle, eigvec/norm(eigvec)
    
    
    def multi_loewner(self, N, q, phi, dphi, sigmaq, thetaq, rvec, lvec):
        '''
        N : number of quadrature nodes
    
        q : number of expected eigenvalues in the contour set
        
        phi : function describing the contour's border
        
        dphi : derivative of phi
    
        sigmaq : right interpolation points
        
        thetaq : left interpolation points
        
        rvec : right interpolation points
        
        lvec : left interpolation vectors
    
        Returns
        -------
        Lamb : eigenvalues
        
        circle: circle in which the contour integration is done
        
        eigvec: normalized eigenvectors
        '''
        n = self.n
        T = self.T
        
        L = zeros((thetaq.size,sigmaq.size), dtype = 'complex_')
        Ls = zeros((thetaq.size,sigmaq.size), dtype = 'complex_')
        Cr = zeros((n,sigmaq.size), dtype = 'complex_')
        Br = zeros((thetaq.size,n), dtype = 'complex_')
        for i in range(0, thetaq.size):
            li = matrix([lvec[:,i]]) 
            li = transpose(li) # n x 1
            bit = zeros((1,n), dtype = 'complex_') # 1 x n
            for k in range(0, N):
                bit += transpose(solve(transpose(T(phi((2 * k * pi) / N))), transpose(li.H * (1 / (1j*N)) * (1 / (thetaq[i] - phi((2 * pi * k) / N))) * dphi((2 * pi * k)/N))))
            Br[i,:] = bit.ravel()
            for j in range(0, sigmaq.size):
                    rj = matrix([rvec[:,j]]) 
                    rj = transpose(rj) # n x 1
                    cj = zeros((n,1), dtype = 'complex_') # n x 1
                    for k in range(0, N):
                        cj += solve(T(phi((2 * k * pi) / N)), rj * (1 / (1j*N)) * (1/ (sigmaq[j] - phi((2 * pi * k) / N))) * dphi((2 * pi * k)/N))
                    L[i,j] = (bit @ rj - li.H @ cj) / (thetaq[i] - sigmaq[j])
                    Ls[i,j] = (thetaq[i] * bit @ rj - sigmaq[j] * li.H @ cj) / (thetaq[i] - sigmaq[j])
                    Cr[:,j] = cj.ravel() 
                
        u, s, vh = svd(L, full_matrices=True)
        sigma = diag(s[0:q]) 
        u = matrix(u[:,0:q])
        vh = matrix(vh[0:q,:])
        B = inv(sigma) @ u.H @ Ls @ vh.H 
        Lamb, Vright = eig(B)
        
        eigvec = Cr @ vh.H @ Vright
        
        x = linspace(0, 2*pi, N)
        circle = phi(x)
        return Lamb, circle, eigvec/norm(eigvec)
    
    def multi1side_loewner(self, N, q, phi, dphi, sigmaq, rvec, lvec):
        '''
        N : number of quadrature nodes
    
        q : number of expected eigenvalues in the contour set
        
        phi : function describing the contour's border
        
        dphi : derivative of phi
    
        sigmaq : right interpolation points
        
        rvec : right interpolation points
        
        lvec : left interpolation vectors
    
        Returns
        -------
        Lamb : eigenvalues
        
        circle: circle in which the contour integration is done
        
        eigvec: normalized eigenvectors
        '''
        n = self.n
        T = self.T
        
        L = zeros((sigmaq.size,sigmaq.size), dtype = 'complex_')
        Ls = zeros((sigmaq.size,sigmaq.size), dtype = 'complex_')
        Cr = zeros((n,sigmaq.size), dtype = 'complex_')
        Br = zeros((sigmaq.size,n), dtype = 'complex_')
        l_dT_r = 0
        for i in range(0, sigmaq.size):
            li = matrix([lvec[:,i]]) 
            li = transpose(li) # n x 1
            bit = zeros((1,n), dtype = 'complex_') # 1 x n
            for k in range(0, N):
                bit += transpose(solve(transpose(T(phi((2 * k * pi) / N))), transpose(li.H * (1 / (1j*N)) * (1 / (sigmaq[i] - phi((2 * pi * k) / N))) * dphi((2 * pi * k)/N))))
            Br[i,:] = bit.ravel()
            for j in range(0, sigmaq.size):
                    rj = matrix([rvec[:,j]]) 
                    rj = transpose(rj) # n x 1
                    cj = zeros((n,1), dtype = 'complex_') # n x 1
                    for k in range(0, N):
                        cj += solve(T(phi((2 * k * pi) / N)), rj * (1 / (1j*N)) * (1/ (sigmaq[j] - phi((2 * pi * k) / N))) * dphi((2 * pi * k)/N))
                    Cr[:,j] = cj.ravel() 
                    if i != j:
                        L[i,j] = (bit @ rj - li.H @ cj) / (sigmaq[i] - sigmaq[j])
                        Ls[i,j] = (sigmaq[i] * bit @ rj - sigmaq[j] * li.H @ cj) / (sigmaq[i] - sigmaq[j])
                    else:
                        l_dT_r = 0
                        for k in range(0, N):
                            l_dT_r = l_dT_r + (1 / (1j*N)) * ((-1) / ((sigmaq[j] - phi((2 * pi * k) / N)) ** 2)) * li.H @ inv(T(phi((2 * k * pi) / N))) @ rj * dphi((2 * pi * k)/N)
                        L[i,j] = l_dT_r
                        Ls[i,j] = (sigmaq[j] * l_dT_r) + (li.H @ cj)
                        
        u, s, vh = svd(L, full_matrices=True)
        sigma = diag(s[0:q]) 
        u = matrix(u[:,0:q])
        vh = matrix(vh[0:q,:])
        B = inv(sigma) @ u.H @ Ls @ vh.H 
        Lamb, Vright = eig(B)
        eigvec = Cr @ vh.H @ Vright
        
        
        x = linspace(0, 2*pi, N)
        circle = phi(x)
        return Lamb, circle, eigvec/norm(eigvec)
    
    
    
    def residualError(self, lamb, v):
        T = self.T
        E = zeros((1,lamb.size), dtype = 'float')
        for i in range (0, lamb.size):
            E[0,i] = norm(T(lamb[i]) @ v[:,i])/(norm(T(lamb[i])))
        return max(E), mean(E, dtype='float')
