import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio

from AgentPrune.graph.node import Node
from AgentPrune.agents.agent_registry import AgentRegistry

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
                                                 requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min() 
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
    
    def update_masks_with_zero_extension(self, pruning_rate: float) -> torch.Tensor:
        """
        使用0-extension算法更新一维掩码进行图剪枝
        
        参数:
        pruning_rate: 要剪枝的边的比例
        
        返回:
        更新后的掩码
        """
        pruner = ZeroExtensionPruner()
        
        # 处理空间掩码 (一维向量)
        if self.optimized_spatial:
            # 1. 提取图的结构 (将一维索引转换为图的顶点和边)
            mask_length = len(self.spatial_masks)
            num_nodes = int(math.sqrt(mask_length))  # 假设一维掩码是从n×n网格展平而来
            
            # 构建顶点集
            vertices = list(range(num_nodes))
            
            # 构建边集 (从一维索引恢复二维关系)
            valid_edge_indices = torch.nonzero(self.spatial_masks, as_tuple=True)[0]
            valid_edges = []
            
            for idx in valid_edge_indices:
                # 将一维索引转换回二维坐标 (i, j)
                i = idx.item() // num_nodes
                j = idx.item() % num_nodes
                if i < num_nodes and j < num_nodes:  # 确保索引有效
                    valid_edges.append((i, j))
            
            # 如果边太少，直接使用原始的topk策略
            if len(valid_edges) < 3:
                # 使用原始的topk方法
                num_edges = (self.spatial_masks > 0).sum()
                num_masks = (self.spatial_masks == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate) > 0 else 1
                _edge_logits = self.spatial_logits.clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.spatial_masks[prune_idx] = 0
                return self.spatial_masks, self.temporal_masks
            
            # 2. 计算每个边的权重 (使用一维logits)
            edge_weights = {}
            for idx, (i, j) in enumerate(valid_edges):
                edge_index = i * num_nodes + j
                if edge_index < len(self.spatial_logits):
                    edge_weights[(i, j)] = float(self.spatial_logits[edge_index])
            
            # 3. 确定终端数量和选择终端
            num_terminals = max(2, min(int(math.sqrt(num_nodes)), len(vertices) // 3))
            num_edges = len(valid_edges)
            prune_num_edges = int(torch.round(torch.tensor(num_edges * pruning_rate)).item()) if torch.round(torch.tensor(num_edges * pruning_rate)) > 0 else 1
            
            terminals = pruner.select_terminals(vertices, valid_edges, edge_weights, num_terminals)
            
            # 4. 运行0-extension算法
            assignment = pruner.zero_extension(vertices, valid_edges, terminals, edge_weights)
            
            # 5. 根据0-extension结果确定要保留的边
            # 只保留连接同一个终端区域内的边或连接终端的边
            new_masks = torch.zeros_like(self.spatial_masks)
            
            for u, v in valid_edges:
                # 如果两个节点属于同一个终端，或者其中之一是终端，则保留该边
                if assignment[u] == assignment[v] or u in terminals or v in terminals:
                    edge_index = u * num_nodes + v
                    if edge_index < len(new_masks):
                        new_masks[edge_index] = 1
            
            # 6. 如果保留的边数量不满足剪枝率，额外剪掉一些边
            kept_edges = (new_masks > 0).sum().item()
            if kept_edges > num_edges - prune_num_edges:
                # 计算还需要剪掉多少边
                extra_prune = kept_edges - (num_edges - prune_num_edges)
                
                # 对保留的边按logits排序，剪掉logits较小的边
                kept_edge_indices = torch.nonzero(new_masks, as_tuple=True)[0]
                kept_edge_logits = torch.tensor([self.spatial_logits[i] for i in kept_edge_indices])
                
                # 分类边：终端间的边和非终端间的边
                terminal_edges = []
                non_terminal_edges = []
                
                for idx in kept_edge_indices:
                    i = idx.item() // num_nodes
                    j = idx.item() % num_nodes
                    if i in terminals and j in terminals:
                        terminal_edges.append(idx.item())
                    else:
                        non_terminal_edges.append(idx.item())
                
                # 如果非终端边不够剪，那么就只能剪掉一部分终端边
                if len(non_terminal_edges) < extra_prune:
                    # 先剪掉所有非终端边
                    for idx in non_terminal_edges:
                        new_masks[idx] = 0
                    
                    # 再剪掉部分终端边
                    remaining_prune = extra_prune - len(non_terminal_edges)
                    terminal_edge_logits = torch.tensor([self.spatial_logits[i] for i in terminal_edges])
                    sorted_indices = torch.argsort(terminal_edge_logits)
                    
                    for i in range(min(remaining_prune, len(sorted_indices))):
                        idx = terminal_edges[sorted_indices[i]]
                        new_masks[idx] = 0
                else:
                    # 只剪掉非终端边
                    non_terminal_edge_logits = torch.tensor([self.spatial_logits[i] for i in non_terminal_edges])
                    sorted_indices = torch.argsort(non_terminal_edge_logits)
                    
                    for i in range(min(extra_prune, len(sorted_indices))):
                        idx = non_terminal_edges[sorted_indices[i]]
                        new_masks[idx] = 0
            
            self.spatial_masks = new_masks
        
        # 处理时间掩码 (类似逻辑)
        if self.optimized_temporal:
            # 时间掩码的处理也改为一维
            mask_length = len(self.temporal_masks)
            num_nodes = int(math.sqrt(mask_length))
            
            vertices = list(range(num_nodes))
            valid_edge_indices = torch.nonzero(self.temporal_masks, as_tuple=True)[0]
            valid_edges = []
            
            for idx in valid_edge_indices:
                i = idx.item() // num_nodes
                j = idx.item() % num_nodes
                if i < num_nodes and j < num_nodes:
                    valid_edges.append((i, j))
            
            # 如果边太少，使用原始topk策略
            if len(valid_edges) < 3:
                num_edges = (self.temporal_masks > 0).sum()
                num_masks = (self.temporal_masks == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate) > 0 else 1
                _edge_logits = self.temporal_logits.clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.temporal_masks[prune_idx] = 0
                return self.spatial_masks, self.temporal_masks
            
            edge_weights = {}
            for i, j in valid_edges:
                edge_index = i * num_nodes + j
                if edge_index < len(self.temporal_logits):
                    edge_weights[(i, j)] = float(self.temporal_logits[edge_index])
            
            num_terminals = max(2, min(int(math.sqrt(num_nodes)), len(vertices) // 3))
            num_edges = len(valid_edges)
            prune_num_edges = int(torch.round(torch.tensor(num_edges * pruning_rate)).item()) if torch.round(torch.tensor(num_edges * pruning_rate)) > 0 else 1
            
            terminals = pruner.select_terminals(vertices, valid_edges, edge_weights, num_terminals)
            assignment = pruner.zero_extension(vertices, valid_edges, terminals, edge_weights)
            
            new_masks = torch.zeros_like(self.temporal_masks)
            for u, v in valid_edges:
                if assignment[u] == assignment[v] or u in terminals or v in terminals:
                    edge_index = u * num_nodes + v
                    if edge_index < len(new_masks):
                        new_masks[edge_index] = 1
            
            kept_edges = (new_masks > 0).sum().item()
            if kept_edges > num_edges - prune_num_edges:
                extra_prune = kept_edges - (num_edges - prune_num_edges)
                
                kept_edge_indices = torch.nonzero(new_masks, as_tuple=True)[0]
                kept_edge_logits = torch.tensor([self.temporal_logits[i] for i in kept_edge_indices])
                
                terminal_edges = []
                non_terminal_edges = []
                for idx in kept_edge_indices:
                    i = idx.item() // num_nodes
                    j = idx.item() % num_nodes
                    if i in terminals and j in terminals:
                        terminal_edges.append(idx.item())
                    else:
                        non_terminal_edges.append(idx.item())
                
                if len(non_terminal_edges) < extra_prune:
                    for idx in non_terminal_edges:
                        new_masks[idx] = 0
                    
                    remaining_prune = extra_prune - len(non_terminal_edges)
                    terminal_edge_logits = torch.tensor([self.temporal_logits[i] for i in terminal_edges])
                    sorted_indices = torch.argsort(terminal_edge_logits)
                    
                    for i in range(min(remaining_prune, len(sorted_indices))):
                        idx = terminal_edges[sorted_indices[i]]
                        new_masks[idx] = 0
                else:
                    non_terminal_edge_logits = torch.tensor([self.temporal_logits[i] for i in non_terminal_edges])
                    sorted_indices = torch.argsort(non_terminal_edge_logits)
                    
                    for i in range(min(extra_prune, len(sorted_indices))):
                        idx = non_terminal_edges[sorted_indices[i]]
                        new_masks[idx] = 0
            
            self.temporal_masks = new_masks
        
        return self.spatial_masks, self.temporal_masks









import math
import random
from collections import defaultdict

class ZeroExtensionPruner:
    def __init__(self):
        """初始化0-extension剪枝器"""
        pass
    
    def compute_semimetric(self, vertices, edges, terminals, edge_weights):
        """
        计算基于边权重的半度量
        
        参数:
        vertices: 所有节点索引列表
        edges: 边的列表[(u,v),...] - 表示为索引对
        terminals: 终端节点索引列表
        edge_weights: 边的权重字典，键为(u,v)，值为权重
        
        返回:
        delta: 半度量字典，键为(u,v)，值为距离
        """
        # 初始化delta为一个大的字典
        delta = defaultdict(lambda: float('inf'))
        
        # 对自己的距离为0
        for v in vertices:
            delta[(v, v)] = 0
        
        # 使用边权重作为初始距离
        for u, v in edges:
            if (u, v) in edge_weights:
                # 使用逆权重作为距离 - 权重大的边距离小
                delta[(u, v)] = 1.0 / (edge_weights[(u, v)] + 1e-6)
                delta[(v, u)] = 1.0 / (edge_weights[(u, v)] + 1e-6)
        
        # 使用Floyd-Warshall算法计算所有节点对之间的最短路径
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if delta[(i, j)] > delta[(i, k)] + delta[(k, j)]:
                        delta[(i, j)] = delta[(i, k)] + delta[(k, j)]
        
        return delta
    
    def zero_extension(self, vertices, edges, terminals, edge_weights, m=None):
        """
        运行0-extension算法
        
        参数:
        vertices: 所有节点索引的列表
        edges: 边的列表[(u,v),...]
        terminals: 终端节点索引列表
        edge_weights: 边的权重字典
        m: 区间分割数，默认为log log k
        
        返回:
        assignment: 每个节点到终端的分配字典
        """
        # 如果未指定m，设置为log log k
        k = len(terminals)
        if m is None:
            m = max(1, int(math.log(math.log(k, 2) + 1, 2)))
        
        # 计算半度量
        delta = self.compute_semimetric(vertices, edges, terminals, edge_weights)
        
        # 找到每个顶点到最近终端的距离
        A = {}
        for v in vertices:
            A[v] = min(delta.get((v, t), float('inf')) for t in terminals)
            if A[v] == float('inf'):
                # 如果节点无法到达任何终端，设为较大但有限的值
                A[v] = 1000.0
        
        # 步骤1: 随机选择γ并四舍五入A值
        gamma = random.uniform(1, 2)
        A_prime = {}
        for v in vertices:
            if v in terminals:  # 终端顶点的A值为0
                A_prime[v] = 0
            else:
                # 四舍五入到下一个2的幂
                power = 0
                while 2**power < 2*A[v]/gamma and power < 30:
                    power += 1
                A_prime[v] = 2**power
        
        # 步骤2: 随机选择区间i, α值和终端的排列
        i = random.randint(1, m)
        alpha = random.uniform(2**(i-1), 2**i)
        terminal_permutation = list(terminals)
        random.shuffle(terminal_permutation)
        
        # 步骤3: 分配顶点到终端
        assignment = {}
        
        # 对于每个终端（按照排列顺序）
        for t in terminal_permutation:
            # 对于每个未分配的顶点
            for v in vertices:
                if v not in assignment:
                    # 如果距离小于阈值，分配顶点到这个终端
                    if delta.get((v, t), float('inf')) <= alpha * A_prime[v]:
                        assignment[v] = t
        
        # 确保所有节点都被分配（以防有节点无法到达任何终端）
        for v in vertices:
            if v not in assignment:
                # 分配到最近的终端
                closest_terminal = min(terminals, key=lambda t: delta.get((v, t), float('inf')))
                assignment[v] = closest_terminal
        
        return assignment

    def select_terminals(self, vertices, edges, edge_weights, num_terminals):
        """
        基于边权重选择终端节点
        
        参数:
        vertices: 所有节点索引的列表
        edges: 边的列表[(u,v),...]
        edge_weights: 边的权重字典
        num_terminals: 要选择的终端数量
        
        返回:
        terminals: 选择的终端节点列表
        """
        # 计算每个节点的重要性 (这里使用连接边的总权重作为简单度量)
        node_importance = defaultdict(float)
        for u, v in edges:
            weight = edge_weights.get((u, v), 0)
            node_importance[u] += weight
            node_importance[v] += weight
        
        # 选择权重最高的节点作为终端
        sorted_vertices = sorted(vertices, key=lambda v: node_importance.get(v, 0), reverse=True)
        terminals = sorted_vertices[:num_terminals]
        
        return terminals
