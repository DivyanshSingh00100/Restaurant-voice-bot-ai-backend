from langgraph.graph import END, START, StateGraph

from app.graphs.conversation_state import ConversationState
from app.graphs.nodes.escalation_handler_node import escalation_handler_node
from app.graphs.nodes.greeter_node import greeter_node
from app.graphs.nodes.menu_handler_node import menu_handler_node
from app.graphs.nodes.order_handler_node import order_handler_node
from app.graphs.routing_edges import route_after_handling, route_initial_turn

def build_conversation_graph():
    graph_builder = StateGraph(ConversationState)

    graph_builder.add_node("greeter", greeter_node)
    graph_builder.add_node("order_handler", order_handler_node)
    graph_builder.add_node("menu_handler", menu_handler_node)
    graph_builder.add_node("escalation_handler", escalation_handler_node)

    graph_builder.add_conditional_edges(
        START,
        route_initial_turn,
        {
            "greeter": "greeter",
            "order_handler": "order_handler",
            "menu_handler": "menu_handler",
            "escalation_handler": "escalation_handler",
        },
    )

    graph_builder.add_edge("greeter", END)

    graph_builder.add_conditional_edges(
        "order_handler",
        route_after_handling,
        {"escalation_handler": "escalation_handler", "end": END},
    )

    graph_builder.add_conditional_edges(
        "menu_handler",
        route_after_handling,
        {"escalation_handler": "escalation_handler", "end": END},
    )

    graph_builder.add_edge("escalation_handler", END)

    return graph_builder.compile()

restaurant_a_graph = build_conversation_graph()