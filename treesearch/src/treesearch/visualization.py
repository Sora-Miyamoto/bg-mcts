from typing import Optional

import graphviz  # type: ignore

from treesearch.algos.tree import Tree

def visualize_tree_paper(
    tree,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    format: str = "pdf",
    nodesep: float=0.1,
    ranksep: float=0.5,
    edge: str="vec",
    budget: float = 30000,
    number: bool = False,
    correct_color: str = "yellow",
    incorrect_color: str = "red",
    correct_shape: str = "star",
    incorrect_shape: str = "triangle",
    penwidth: str = "3",
    stopping: bool = False,
):

    try:
        dot = graphviz.Digraph(comment=title or "Tree Visualization")
        
        if title:
            dot.attr(label=title)
        
        dot.attr('node', colorscheme="blues9", style="filled", margin='0')
        dot.attr(nodesep=str(nodesep))
        dot.attr(ranksep=str(ranksep))  
        dot.attr(margin='0')
        dot.attr(overlap='true')
        dot.attr('node', shape='circle', fixedsize='true', width='0.35', height='0.35', fontsize='10')
        dot.attr('edge', arrowhead=edge) 
        # Add nodes and edges
        nodes = tree.tree.get_nodes()
        node_num = len(nodes)
        
        for node in nodes:
            node_id = str(node.expand_idx)
            if node.total_cost > budget:
                break
            if node.is_root():
                label = f"Root"
                color = "1"
                current_scheme = "blues9"
                dot.node(node_id, label=label, style="filled", colorscheme=current_scheme, fillcolor=color, shape="circle")
                if node.parent:
                    dot.edge(str(node.parent.expand_idx), node_id)
            else:
                llm_ans = node.state.generation_result.generation_state
                if number:
                    label = f"{node.expand_idx + 1}"
                else:
                    label = ""
                if llm_ans[0] == "process":
                    color_idx= int(6 * ((node.total_cost) / budget)) + 2
                    color = str(color_idx)
                    current_scheme = "blues9"
                    # is_special = False
                    dot.node(node_id, label=label, style="filled", colorscheme=current_scheme, fillcolor=color, shape="circle")
                    if node.parent:
                        dot.edge(str(node.parent.expand_idx), node_id)
                elif llm_ans[0] == "correct":
                    color_idx= int(6 * ((node.total_cost) / budget)) + 2
                    color = str(color_idx)
                    current_scheme = "blues9"
                    dot.node(node_id, label=label, style="filled", colorscheme="", fillcolor=f"/blues9/{color_idx}", color=correct_color, penwidth=penwidth, shape=correct_shape)
                    if node.parent:
                        dot.edge(str(node.parent.expand_idx), node_id)
                    if stopping and node.score >= 0.9:
                        print("Early Stop")
                        break
                elif llm_ans[0] == "incorrect":
                    color_idx= int(6 * ((node.total_cost) / budget)) + 2
                    color = str(color_idx)
                    current_scheme = "blues9"
                    dot.node(node_id, label=label, style="filled", colorscheme="", fillcolor=f"/blues9/{color_idx}", color=incorrect_color, penwidth=penwidth, shape=incorrect_shape)
                    if node.parent:
                        dot.edge(str(node.parent.expand_idx), node_id)
                    if stopping and node.score >= 0.9:
                        print("Early Stop")
                        break
        if save_path:
            dot.render(save_path, format=format, cleanup=True)

        return dot

    except graphviz.backend.execute.ExecutableNotFound:
        print("graphviz executable is not in system Path, visualization skipped...")
        return None

