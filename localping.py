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
        request = search_rec_pb2.SearchRequest(input_query="dog token")
        
        # Create metadata with the token
        metadata = [('authorization', f'Bearer {token}')]

        # Make the call
        try:
            response = stub.Search(request, metadata=metadata)
            print("Search Response:")
            print(f"Answer: {response.answer}")
            for item in response.items[:3]:
                print(f"Canister ID: {item.canister_id}")
                print(f"Description: {item.description}")
                print(f"Host: {item.host}")
                print(f"Link: {item.link}")
                print(f"Logo: {item.logo}")
                print(f"Token Name: {item.token_name}")
                print(f"Token Symbol: {item.token_symbol}")
                print(f"User ID: {item.user_id}")
                print(f"Created At: {item.created_at}")
                print("-" * 20)
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()