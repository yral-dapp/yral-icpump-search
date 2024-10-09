import os
import grpc
import search_rec_pb2
import search_rec_pb2_grpc
# server = 'localhost:50052'
server = 'yral-icpumpsearch.fly.dev:443'

def run():
    # Read the token from the environment variable
    token = os.getenv('SEARCH_TOKEN_ICPUMP')
    

    with grpc.secure_channel(server, credentials=grpc.ssl_channel_credentials()) as channel:
        stub = search_rec_pb2_grpc.SearchServiceStub(channel)
        request = search_rec_pb2.SearchRequest(input_query="what are some tokens related to dog? What are they talking about?")
        
        
        response = stub.Search(request)#, metadata=metadata)
        print("Search service is up and running!")
        print("Received response:")
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

if __name__ == '__main__':
    run()