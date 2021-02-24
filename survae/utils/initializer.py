from jax import numpy as jnp, random  
import jax

def rvs(rng, dim): 
    H = jnp.eye(dim)
    D = jnp.ones((dim,))
    for n in range(1, dim):
        x = random.normal(rng, shape=(dim-n+1,))
        # D[n-1] = jnp.sign(x[0])
        D = jax.ops.index_update(D,n-1,jnp.sign(x[0])) 
        # x[0] -= D[n-1]*jnp.sqrt((x*x).sum())
        x = jax.ops.index_add(x,0,-D[n-1]*jnp.sqrt((x*x).sum()))
        # Householder transformation
        Hx = (jnp.eye(dim-n+1) - 2.*jnp.outer(x, x)/(x*x).sum())
        mat = jnp.eye(dim)
        # mat[n-1:, n-1:] = Hx
        mat = jax.ops.index_update(mat,jax.ops.index[n-1:, n-1:],Hx)
        H = jnp.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    # D[-1] = (-1)**(1-(dim % 2))*D.prod()
    D = jax.ops.index_update(D,-1,(-1)**(1-(dim % 2))*D.prod())
    # Equivalent to np.dot(jnp.diag(D), H) but faster, apparently
    H = (D*H.T).T
    H = jax.lax.stop_gradient(H)
    return H