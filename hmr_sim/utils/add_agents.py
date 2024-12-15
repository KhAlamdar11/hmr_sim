import numpy as np
from hmr_sim.utils.utils import svstack, distance
from hmr_sim.utils.agent import Agent
from copy import deepcopy


class AddAgent:

    def __init__(self, params, agents, comm_radius):

        # add_agent_base, add_agent_near
        self.mode = params['mode']

        # In hetrogenous systems, specify which kind of agent you are adding
        self.type = params['agent_type_to_add']

        # For Mode: add_agent_base -> ID of the agent where the new agent must be added
        self.agent_addition_id = params['agent_addition_id']

        self.agents = agents        
        self.comm_radius = comm_radius

        # internal add agents params
        self.n_samples_per = 15

        # For add_agent_near only: Neighboring agents
        self.neighbors = None


        # Dispatch table for mode-to-method mapping
        self.mode_dispatch = {
            'add_agent_base': self.add_agent_base,
            'add_agent_near': self.add_agent_near,
        }


    def __call__(self):
        if self.mode in self.mode_dispatch:
            self.mode_dispatch[self.mode]()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    def add_agent_base(self):
        
        # Find base position
        pos = self.find_position()
        print(f"base: {pos}")

        # Add new agent at position
        new_pos = self.add_agent_at(pos)
        print(f"Adding new agent at position: {new_pos}")

        self.create_new_agent(new_pos)


    def add_agent_near(self):
        
        arc = np.array([])
        poss = np.array([])

        for neigh in self.neighbors:
            neigh_pos = neigh.get_pos()
            res = self.add_agent_at(neigh_pos,return_all=True,factor_comm=0.37,factor_dist=0.3)
            if res is None:
                continue
            a, pos = res
            print(a,pos)
            print("poss",poss)
            arc = svstack([arc, a])
            poss = np.concatenate([poss,pos])


        if poss.shape[0]!=0:
            m = np.argmax(poss)
            
            new_pos = np.array([arc[m,0],arc[m,1]])

            self.create_new_agent(new_pos)


            


    #______________________  utils  _____________________

    def find_sample_agent(self):
        for agent in self.agents:
            if agent.type == self.type:
                return agent

    def find_position(self):
        for agent in self.agents:
            if agent.agent_id == self.agent_addition_id:
                return agent.get_pos()

    def set_neighbors(self,neighbors):
        self.neighbors = neighbors

    def add_agent_at(self,pos,return_all=False, factor_comm=0.7, factor_dist=0.7):
        '''
        Main function for add agent base strategy. Adds a new agent at the base.
        '''
        t = np.linspace(0, 2 * np.pi, self.n_samples_per)   
        a, b = factor_comm*self.comm_radius*np.cos(t), factor_comm*self.comm_radius*np.sin(t)
        a += pos[0]
        b += pos[1]
        arc = np.array([[a, b] for a, b in zip(a, b)])

        # shortlist pts
        possible_pts = []
        possible_connections = []
        for j in range(arc.shape[0]):
            allow = True
            cons = 0
            for agent in self.agents:
                if distance(arc[j],agent.get_pos()) < self.comm_radius*factor_dist:
                    allow = False
                    break
                elif agent.agent_id != self.agent_addition_id and\
                                    distance(arc[j],agent.get_pos()) < self.comm_radius:
                    if self.mode == 'add_agent_base':
                        cons += 1
                    elif self.mode == 'add_agent_near' and agent in self.neighbors:
                        cons += 1
            if allow:
                possible_pts.append(arc[j])
                possible_connections.append(cons)

        if len(possible_connections) != 0:
            arc, poss = np.array(possible_pts), np.array(possible_connections)
            m = np.argmax(poss)
        else:
            return None
        
        if return_all:
            return np.array(possible_pts), np.array(possible_connections)

        return np.array([arc[m,0],arc[m,1]])



    def add_agent_i(self,agent,n):
        '''
        Helper function for add agent near strategy.
        '''
        t = np.linspace(0, 2 * np.pi, 9)   
        a, b = 0.4*self.comm_radius*np.cos(t), 0.4*self.comm_radius*np.sin(t)
        a += self.x[n,0]
        b += self.x[n,1]
        arc = np.array([[a, b] for a, b in zip(a, b)])

        # shortlist pts
        possible_pts = []
        possible_connections = []
        for j in range(arc.shape[0]):
            allow = True
            cons = 0
            for i in range(self.n_agents):
                if i != n and distance(arc[j],self.x[i]) < self.comm_radius*0.3:
                    allow = False
                    break
                elif i != n and i!=agent and distance(arc[j],self.x[i]) < self.comm_radius :
                    cons += 1
            if allow:
                possible_pts.append(arc[j])
                possible_connections.append(cons)

        return np.array(possible_pts), np.array(possible_connections)


    def create_new_agent(self,pos):
        # Create copy of a agent type of same name, and reinitialize its position
        agent = self.find_sample_agent()
        new_agent = deepcopy(agent)
        new_agent.battery = 1.0
        new_agent.set_pos(pos)
        new_agent.old_path = []
        self.agents.append(new_agent)


# def add_agent(add_agent_mode, agents, total_agents):
#     if add_agent_mode == 'add_agent_base':
#         pos = np.array([0.0,0.0])
#         add_agent_pos(1,pos,agents,total_agents)


# def add_agent_pos(type,pos,agents,total_agents):
#     '''
#     Adds a new agent at a certain position pos
#     '''
#     for agent in agents:
#         if agent.type == 0:
#             # print(f"Adding new agent at {pos}.")
#             new_agent = deepcopy(agent)
#             new_agent.set_pos(pos)
#             new_agent.battery = 1.0
#             new_agent.agent_id = total_agents
#             agents.append(new_agent)
#             return

# def add_agent(self,agent):
#     '''
#     Main function for add agent near strategy. Adds a new agent in the vicinity of the depleting agent.
#     '''
#     neighbors = np.nonzero(self.state_network[agent, :])[0]
#     arc = np.array([])
#     poss = np.array([])
#     for neigh in neighbors:
#         a, pos = self.add_agent_i(agent,neigh)
#         arc = svstack([arc, a])
#         poss = np.concatenate([poss,pos])
    
#     if poss.shape[0]!=0:
#         m = np.argmax(poss)
        
#         new_agent = np.array([arc[m,0],arc[m,1],0,0])

#         self.x = np.vstack([self.x,new_agent])
#         self.n_agents += 1
#         self.battery = np.append(self.battery, 1.0)
#         self.in_motion = np.append(self.in_motion, False)


# def add_agent_i(self,agent,n):
#     '''
#     Helper function for add agent near strategy.
#     '''
#     t = np.linspace(0, 2 * np.pi, 9)   
#     a, b = 0.4*self.comm_radius*np.cos(t), 0.4*self.comm_radius*np.sin(t)
#     a += self.x[n,0]
#     b += self.x[n,1]
#     arc = np.array([[a, b] for a, b in zip(a, b)])

#     # shortlist pts
#     possible_pts = []
#     possible_connections = []
#     for j in range(arc.shape[0]):
#         allow = True
#         cons = 0
#         for i in range(self.n_agents):
#             if i != n and distance(arc[j],self.x[i]) < self.comm_radius*0.3:
#                 allow = False
#                 break
#             elif i != n and i!=agent and distance(arc[j],self.x[i]) < self.comm_radius :
#                 cons += 1
#         if allow:
#             possible_pts.append(arc[j])
#             possible_connections.append(cons)

#     return np.array(possible_pts), np.array(possible_connections)



# def add_agent_base(self,agent):
#     '''
#     Main function for add agent base strategy. Adds a new agent at the base.
#     '''
#     t = np.linspace(0, 2 * np.pi, 15)   
#     a, b = 0.7*self.comm_radius*np.cos(t), 0.7*self.comm_radius*np.sin(t)
#     a += self.x[0,0]
#     b += self.x[0,1]
#     arc = np.array([[a, b] for a, b in zip(a, b)])

#     # shortlist pts
#     possible_pts = []
#     possible_connections = []
#     for j in range(arc.shape[0]):
#         allow = True
#         cons = 0
#         for i in range(self.x.shape[0]):
#             if i != 0 and distance(arc[j],self.x[i]) < self.comm_radius*0.7:
#                 allow = False
#                 break
#             elif i != 0 and i!=agent and distance(arc[j],self.x[i]) < self.comm_radius:
#                 cons += 1
#         if allow:
#             possible_pts.append(arc[j])
#             possible_connections.append(cons)

#     arc, poss = np.array(possible_pts), np.array(possible_connections)
#     m = np.argmax(poss)
    
#     new_agent = np.array([arc[m,0],arc[m,1],0,0])

#     self.x = np.vstack([self.x,new_agent])
#     self.n_agents += 1
#     self.battery = np.append(self.battery, 1.0)
#     self.in_motion = np.append(self.in_motion, False)