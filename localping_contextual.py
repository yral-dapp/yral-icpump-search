import os
import grpc
import search_rec_pb2
import search_rec_pb2_grpc

def run_contextual_search():
    # Read the token from the environment variable
    token = os.getenv('SEARCH_TOKEN_ICPUMP')

    # Connect to the server with an insecure channel
    with grpc.insecure_channel('localhost:50051') as channel:
        # Create a stub (client)
        stub = search_rec_pb2_grpc.SearchServiceStub(channel)
        
        # Create metadata with the token
        metadata = [('authorization', f'Bearer {token}')]

        # Simulate a conversation
        conversation = [
            ("What is the Fibonacci sequence?", "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. For example: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, and so on."),
            ("Can you show me how to calculate it?", "Certainly! Here's a simple explanation of how to calculate the Fibonacci sequence:\n\n1. Start with 0 and 1 as the first two numbers.\n2. To get the next number, add the two previous numbers.\n3. Repeat step 2 to continue the sequence.\n\nFor example:\n0 + 1 = 1\n1 + 1 = 2\n1 + 2 = 3\n2 + 3 = 5\n3 + 5 = 8\nAnd so on...")
        ]

        # Convert conversation to QueryResponsePair objects
        previous_interactions = [
            search_rec_pb2.QueryResponsePair(query=q, response=r)
            for q, r in conversation
        ]

        # Create a request for contextual search
        request = search_rec_pb2.ContextualSearchRequest(
            input_query="Now, can you write a Python function for this?",
            previous_interactions=previous_interactions,
            rag_data=""  # Add any relevant RAG data if available
        )

        # Make the call
        try:
            response = stub.ContextualSearch(request, metadata=metadata)
            print("Contextual Search Response:")
            print(f"Answer: {response.answer}")
            print(f"RAG Data: {response.rag_data}")
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run_contextual_search()