#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
from soap_jax import soap

class Equation:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError

class Boundless_flow(Equation):
    def __init__(self, all_params):
        self.all_params = all_params
    
    @staticmethod
    def Loss(dynamic_params, all_params, g_batch, gv_batch, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        _, out_x = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        _, out_y = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        _, out_z = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
                                                                        

        u = out_y[:,2:3] - out_z[:,1:2]
        v = out_z[:,0:1] - out_x[:,2:3]
        w = out_x[:,1:2] - out_y[:,0:1]
        

        loss_u = u - gv_batch[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = v - gv_batch[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = w - gv_batch[:,2:3]
        loss_w = jnp.mean(loss_w**2)


        total_loss = loss_u + loss_v + loss_w
        return total_loss
    
    @staticmethod
    def Loss_report(dynamic_params, all_params, g_batch, gv_batch, model_fns):
        def first_order(all_params, g_batch, cotangent, model_fns):
            def u_t(batch):
                return model_fns(all_params, batch)
            out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
            return out, out_t

        all_params["network"]["layers"] = dynamic_params
        weights = all_params["problem"]["loss_weights"]
        out, out_t = first_order(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        _, out_x = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        _, out_y = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
        _, out_z = first_order(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)
                                                                        

        u = out_y[:,2:3] - out_z[:,1:2]
        v = out_z[:,0:1] - out_x[:,2:3]
        w = out_x[:,1:2] - out_y[:,0:1]
        

        loss_u = u - gv_batch[:,0:1]
        loss_u = jnp.mean(loss_u**2)

        loss_v = v - gv_batch[:,1:2]
        loss_v = jnp.mean(loss_v**2)

        loss_w = w - gv_batch[:,2:3]
        loss_w = jnp.mean(loss_w**2)

        total_loss = loss_u + loss_v + loss_w
        return total_loss, loss_u, loss_v, loss_w
