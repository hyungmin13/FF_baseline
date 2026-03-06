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
from scipy.spatial import KDTree
import itertools
import pyff3
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

@partial(jax.jit, static_argnums=(1, 2, 5, 8))
def PINN_update(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, grids, grid_v, model_fn):
    static_leaves, treedef = static_keys
    leaves = [d if s is None else s for d, s in zip(static_params, static_leaves)]
    all_params = jax.tree_util.tree_unflatten(treedef, leaves)
    lossval, grads = value_and_grad(equation_fn, argnums=0)(dynamic_params, all_params, grids, grid_v, model_fn)
    updates, model_states = optimiser_fn(grads, model_states, dynamic_params)
    dynamic_params = optax.apply_updates(dynamic_params, updates)
    return lossval, model_states, dynamic_params

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c

class PINN(PINNbase):
    def train(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}, "flowfit":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        key, network_key = random.split(global_key)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        all_params['flowfit'] = self.c.flowfit.init_params(**self.c.flowfit_init_kwargs)
        # Initialize optmiser
        learn_rate = optax.exponential_decay(self.c.optimization_init_kwargs["learning_rate"],
                                             self.c.optimization_init_kwargs["decay_step"],
                                             self.c.optimization_init_kwargs["decay_rate"],)
        optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate, b1=0.95, b2=0.95,
                                                                 weight_decay=0.01, precondition_frequency=5)
        #optimiser = self.c.optimization_init_kwargs["optimiser"](learning_rate=learn_rate)
        model_states = optimiser.init(all_params["network"]["layers"])
        optimiser_fn = optimiser.update
        model_fn = c.network.network_fn
        dynamic_params = all_params["network"].pop("layers")

        # Define equation function
        equation_fn = self.c.equation.Loss
        report_fn = self.c.equation.Loss_report
        flowfit_fn = self.c.flowfit.FF3_python
        # Input data and grids
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        valid_data = self.c.problem.exact_solution(all_params.copy())

        try:
            downsample = str(all_params['data']['path'][all_params['data']['path'].find('lv')+2:all_params['data']['path'].find('lv')+4])
        except:
            downsample = str(all_params['data']['path'][all_params['data']['path'].find('lv')+2:all_params['data']['path'].find('lv')+3])
        _, counts = np.unique(train_data['pos'][:,0], return_counts=True)
        if os.path.isdir(cur_dir + '/' + 'ff_coeff/lv' + str(downsample)) == False:
            os.mkdir(cur_dir + '/' + 'ff_coeff/lv' + str(downsample))
            for i in range(counts.shape[0]):
                flowfit_fn(all_params['domain']['domain_range']['x'][0], all_params['domain']['domain_range']['x'][1],
                        all_params['domain']['domain_range']['y'][0], all_params['domain']['domain_range']['y'][1],
                        all_params['domain']['domain_range']['z'][0], all_params['domain']['domain_range']['z'][1],
                        all_params['flowfit']['h'], all_params['flowfit']['hf'], all_params['flowfit']['ep'], downsample,
                        train_data['pos'][sum(counts[:i]):sum(counts[:i+1]),1:], 
                        train_data['vel'][sum(counts[:i]):sum(counts[:i+1]),:], 
                        i)
        else:
            print("FF coeffs already exist, skipping FF coeff generation.")
        #model_states = optimiser.init(all_params["network"]["layers"])
        #optimiser_fn = optimiser.update
        #model_fn = c.network.network_fn
        #dynamic_params = all_params["network"].pop("layers")

        # Input key initialization
        key, batch_key = random.split(key)
        num_keysplit = 10
        keys = random.split(batch_key, num = num_keysplit)
        keys_split = [random.split(keys[i], num = self.c.optimization_init_kwargs["n_steps"]) for i in range(num_keysplit)]
        keys_iter = [iter(keys_split[i]) for i in range(num_keysplit)]
        keys_next = [next(keys_iter[i]) for i in range(num_keysplit)]
        print(len(keys_iter))
        print(len(keys_next))
        # Static parameters
        leaves, treedef = jax.tree_util.tree_flatten(all_params)
        static_params = tuple(x if isinstance(x,(np.ndarray, jnp.ndarray)) else None for x in leaves)
        static_leaves = tuple(None if isinstance(x,(np.ndarray, jnp.ndarray)) else x for x in leaves)
        static_keys = (static_leaves, treedef)
        
        # Initializing batches
        g_batch = np.stack([random.choice(keys_next[k+1], 
                                           grids['eqns'][arg], 
                                           shape=(self.c.optimization_init_kwargs["e_batch"]*1000,)) 
                             for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)

        g_batch = g_batch[np.lexsort((g_batch[:,0],))]
        g_batch[:,1] = g_batch[:,1]*all_params["domain"]["domain_range"]["x"][1]
        g_batch[:,2] = g_batch[:,2]*all_params["domain"]["domain_range"]["y"][1]
        g_batch[:,3] = g_batch[:,3]*all_params["domain"]["domain_range"]["z"][1]
        _, g_count = np.unique(g_batch[:,0], return_counts=True)
        vfield_names = np.sort(glob(cur_dir + '/' + 'ff_coeff/lv' + str(downsample) + '/*.hdf5'))
        gv_batch = []

        for i in range(len(vfield_names)):
            print(vfield_names[i])
            vfield = pyff3.VelocityField(vfield_names[i])
            vfield_result = vfield.sample_at(g_batch[np.sum(g_count[:i]):np.sum(g_count[:i+1]),1:])
            loc, vel, _ = vfield_result.data()
            gv_batch.append(vel)

        gv_batch = jnp.concatenate(gv_batch,axis=0)

        perm_g = random.permutation(keys_next[0], g_batch.shape[0])
        data_g = []
        data_gv = []
        for i in range(g_batch.shape[0]//self.c.optimization_init_kwargs["e_batch"]):
            batch_g = g_batch[perm_g[i*self.c.optimization_init_kwargs["e_batch"]:(i+1)*self.c.optimization_init_kwargs["e_batch"]],:]
            batch_gv = gv_batch[perm_g[i*self.c.optimization_init_kwargs["e_batch"]:(i+1)*self.c.optimization_init_kwargs["e_batch"]],:]
            data_g.append(batch_g)
            data_gv.append(batch_gv)
        #data_g.append(g_batch[perm_g[(i+1)*self.c.optimization_init_kwargs["e_batch"]:],:])
        #data_gv.append(gv_batch[perm_g[(i+1)*self.c.optimization_init_kwargs["e_batch"]:],:])
        ss = g_batch.copy()
        g_batches = itertools.cycle(data_g)
        gv_batches = itertools.cycle(data_gv)
        batch_g = next(g_batches)

        batch_gv = next(gv_batches)

        # Initializing the update function
        update = PINN_update.lower(model_states, optimiser_fn, equation_fn, dynamic_params, static_params, static_keys, batch_g, batch_gv, model_fn).compile()
 

        # Training loop
        for i in range(self.c.optimization_init_kwargs["n_steps"]):
            if i != 0:
                keys_next = [next(keys_iter[i]) for i in range(10)]
                g_batch = np.stack([random.choice(keys_next[k+1], 
                                                grids['eqns'][arg], 
                                                shape=(self.c.optimization_init_kwargs["e_batch"]*1000,)) 
                                    for k, arg in enumerate(list(all_params["domain"]["domain_range"].keys()))],axis=1)
                
                g_batch = g_batch[np.lexsort((g_batch[:,0],))]
                g_batch[:,1] = g_batch[:,1]*all_params["domain"]["domain_range"]["x"][1]
                g_batch[:,2] = g_batch[:,2]*all_params["domain"]["domain_range"]["y"][1]
                g_batch[:,3] = g_batch[:,3]*all_params["domain"]["domain_range"]["z"][1]
                _, g_count = np.unique(g_batch[:,0], return_counts=True)

                gv_batch = []
                gv_batch = []

                for k in range(len(vfield_names)):
                    vfield = pyff3.VelocityField(vfield_names[k])
                    vfield_result = vfield.sample_at(g_batch[np.sum(g_count[:k]):np.sum(g_count[:k+1]),1:])
                    loc, vel, _ = vfield_result.data()
                    gv_batch.append(vel)

                gv_batch = jnp.concatenate(gv_batch,axis=0)

                perm_g = random.permutation(keys_next[0], g_batch.shape[0])
                data_g = []
                data_gv = []
                for k in range(g_batch.shape[0]//self.c.optimization_init_kwargs["e_batch"]):
                    batch_g = g_batch[perm_g[k*self.c.optimization_init_kwargs["e_batch"]:(k+1)*self.c.optimization_init_kwargs["e_batch"]],:]
                    batch_gv = gv_batch[perm_g[k*self.c.optimization_init_kwargs["e_batch"]:(k+1)*self.c.optimization_init_kwargs["e_batch"]],:]
                    data_g.append(batch_g)
                    data_gv.append(batch_gv)

            for j in range(self.c.optimization_init_kwargs["save_step"]):
                batch_g = next(g_batches)
                batch_gv = next(gv_batches)
                lossval, model_states, dynamic_params = update(model_states, dynamic_params, static_params, batch_g, batch_gv)
            
                self.report(i*self.c.optimization_init_kwargs["save_step"]+j, report_fn, dynamic_params, all_params, batch_g, batch_gv, valid_data, keys_iter[-1], self.c.optimization_init_kwargs["save_step"], model_fn)
                self.save_model(i*self.c.optimization_init_kwargs["save_step"]+j, dynamic_params, all_params, self.c.optimization_init_kwargs["save_step"], model_fn)

    def save_model(self, i, dynamic_params, all_params, save_step, model_fns):
        model_save = (i % save_step == 0)
        if model_save:
            all_params["network"]["layers"] = dynamic_params
            model = Model(all_params["network"]["layers"], model_fns)
            serialised_model = to_state_dict(model)
            with open(self.c.model_out_dir + "saved_dic_"+str(i)+".pkl","wb") as f:
                pickle.dump(serialised_model,f)
        return
    
    def cal_grad(self, all_params, g_batch, cotangent, model_fns):
        def u_t(batch):
            return model_fns(all_params, batch)
        out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
        return out, out_t
    
    def report(self, i, report_fn, dynamic_params, all_params, batch_g, batch_gv, valid_data, e_batch_key, save_step, model_fns):
        save_report = (i % save_step == 0)
        if save_report:
            all_params["network"]["layers"] = dynamic_params
            e_key = next(e_batch_key)
            e_batch_pos = random.choice(e_key, valid_data['pos'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            e_batch_pos[:,1] = e_batch_pos[:,1]*all_params["domain"]["domain_range"]["x"][1]
            e_batch_pos[:,2] = e_batch_pos[:,2]*all_params["domain"]["domain_range"]["y"][1]
            e_batch_pos[:,3] = e_batch_pos[:,3]*all_params["domain"]["domain_range"]["z"][1]
            e_batch_vel = random.choice(e_key, valid_data['vel'], shape = (self.c.optimization_init_kwargs["e_batch"],))
            _, out_x = self.cal_grad(all_params, e_batch_pos, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(e_batch_pos.shape[0],1)), model_fns)
            _, out_y = self.cal_grad(all_params, e_batch_pos, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(e_batch_pos.shape[0],1)), model_fns)
            _, out_z = self.cal_grad(all_params, e_batch_pos, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(e_batch_pos.shape[0],1)), model_fns)
            u_pred = out_y[:,2:3] - out_z[:,1:2]
            v_pred = out_z[:,0:1] - out_x[:,2:3]
            w_pred = out_x[:,1:2] - out_y[:,0:1]


            u_error = jnp.sqrt(jnp.mean((u_pred - e_batch_vel[:,0:1])**2)/jnp.mean(e_batch_vel[:,0:1]**2))
            v_error = jnp.sqrt(jnp.mean((v_pred - e_batch_vel[:,1:2])**2)/jnp.mean(e_batch_vel[:,1:2]**2))
            w_error = jnp.sqrt(jnp.mean((w_pred - e_batch_vel[:,2:3])**2)/jnp.mean(e_batch_vel[:,2:3]**2))
            #if v_pred.shape[1] == 5:
            #    T_error = jnp.sqrt(jnp.mean((all_params["data"]["T_ref"]*v_pred[:,4] - e_batch_T)**2)/jnp.mean(e_batch_T**2))

            Losses = report_fn(dynamic_params, all_params, batch_g, batch_gv, model_fns)
            if v_pred.shape[1] == 5:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}}\n")
            
            else:
                print(f"step_num : {i:<{12}} u_loss : {Losses[1]:<{12}.{5}} v_loss : {Losses[2]:<{12}.{5}} w_loss : {Losses[3]:<{12}.{5}} u_error : {u_error:<{12}.{5}} v_error : {v_error:<{12}.{5}} w_error : {w_error:<{12}.{5}}")
                with open(self.c.report_out_dir + "reports.txt", "a") as f:
                    f.write(f"{i:<{12}} {Losses[0]:<{12}.{5}} {Losses[1]:<{12}.{5}} {Losses[2]:<{12}.{5}} {Losses[3]:<{12}.{5}} {u_error:<{12}.{5}} {v_error:<{12}.{5}} {w_error:<{12}.{5}} {0.0:<{12}.{5}}\n")
            f.close()

        return

#%%
if __name__=="__main__":
    from domain import *
    from trackdata import *
    from network import *
    from constants import *
    from problem import *
    from equation import *
    from txt_reader import *
    import argparse
    
    parser = argparse.ArgumentParser(description='TBL_PINN')
    parser.add_argument('-n', '--name', type=str, help='run name', default='HIT_k1')
    parser.add_argument('-c', '--config', type=str, help='configuration', default='test_txt')
    args = parser.parse_args()
    cur_dir = os.getcwd()
    input_txt = cur_dir + '/' + args.config + '.txt' 
    data = parse_tree_structured_txt(input_txt)
    c = Constants(**data)

    run = PINN(c)
    run.train()