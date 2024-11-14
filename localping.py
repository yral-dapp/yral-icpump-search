import os
import grpc
import search_rec_pb2
import search_rec_pb2_grpc

def run():
    # Read the token from the environment variable
    token = os.getenv('SEARCH_TOKEN_ICPUMP')

    # Connect to the server with an insecure channel
    with grpc.insecure_channel('localhost:50051') as channel:
        # Create a stub (client)
        stub = search_rec_pb2_grpc.SearchServiceStub(channel)
        
        # Create a request
        request = search_rec_pb2.SearchRequest(input_query="write a code for fibonacci sequence")
        # request = search_rec_pb2.SearchRequest(input_query="what are tokens related to dogs?")
        
        # Create metadata with the token
        metadata = [('authorization', f'Bearer {token}')]

        # Make the call
        try:
            response = stub.SearchV1(request, metadata=metadata)
            print("Search Response:")
            print(f"Answer: {response.answer}")
            print('\n--XX--\n')
            rag_data = response.rag_data 
            
            ## follow ups 
            follow_up_query = "what is the token symbol of the token related to dogs?"
            follow_up_request = search_rec_pb2.ContextualSearchRequest(input_query=follow_up_query, previous_interactions=[], rag_data=rag_data)
            follow_up_response = stub.ContextualSearch(follow_up_request, metadata=metadata)
            
            next_query = "what is the token name of the token related to dogs?"
            next_request = search_rec_pb2.ContextualSearchRequest(input_query=next_query, previous_interactions=[{"query": follow_up_query, "response": follow_up_response.answer}], rag_data=rag_data)
            next_response = stub.ContextualSearch(next_request, metadata=metadata)
            
            print("Follow-up Response:")
            print(f"Answer: {follow_up_response.answer}")
            print("Next follow up response:")
            print(f"Answer: {next_response.answer}")
            print("Next follow up response:")
            print('--XX--')
            
            
            # for item in response.items[:3]:
            #     print(f"Canister ID: {item.canister_id}")
            #     print(f"Description: {item.description}")
            #     print(f"Host: {item.host}")
            #     print(f"Link: {item.link}")
            #     print(f"Logo: {item.logo}")
            #     print(f"Token Name: {item.token_name}")
            #     print(f"Token Symbol: {item.token_symbol}")
            #     print(f"User ID: {item.user_id}")
            #     print(f"Created At: {item.created_at}")
            #     print("-" * 20)

        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()