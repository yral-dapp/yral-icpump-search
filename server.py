from concurrent import futures
import jwt
import logging
import multiprocessing
import sys
import time
import pickle
import grpc
from grpc_reflection.v1alpha import reflection
import consts
import search_rec_pb2
import search_rec_pb2_grpc
from grpc_interceptor import ServerInterceptor
from grpc_interceptor.exceptions import GrpcException
from collections import defaultdict

from search_agent_bq import SearchAgent

_ONE_DAY = 24 * 60 * 60
_PROCESS_COUNT = multiprocessing.cpu_count()
# _PROCESS_COUNT = 1
_THREAD_CONCURRENCY = 10
_BIND_ADDRESS = "[::]:50051"
_AUTH_HEADER_KEY = "authorization"
_PUBLIC_KEY = consts.SEARCH_JWT_PUBLIC_KEY
_JWT_PAYLOAD = {
    "sub": "yral-icpump-search",
    "company": "gobazzinga",
}

_LOGGER = logging.getLogger(__name__)

class SearchServicer(search_rec_pb2_grpc.SearchServiceServicer):
    def __init__(self):
        self.search_agent = SearchAgent()  # TODO: turn debug back to false

    def Search(self, request, context):
        search_query = request.input_query
        # _LOGGER.info(f"Received search query: {search_query}")
        response = search_rec_pb2.SearchResponse()
        try:
            df, answer, rag_data = self.search_agent.process_query(search_query)
            response.answer = answer
            total_responses_fetched = len(df)
            for i in range(total_responses_fetched):
                item = response.items.add()
                item.canister_id = df.iloc[i]['canister_id']
                item.description = df.iloc[i]['description']
                item.host = df.iloc[i]['host']
                item.link = df.iloc[i]['link']
                item.logo = df.iloc[i]['logo']
                item.token_name = df.iloc[i]['token_name']
                item.token_symbol = df.iloc[i]['token_symbol']
                item.user_id = df.iloc[i]['user_id']
                item.created_at = df.iloc[i]['created_at']
            response.rag_data = rag_data
        except Exception as e:
            _LOGGER.error(f"SearchAgent failed: {e}")
        return response

    def ContextualSearch(self, request, context):
        input_query = request.input_query
        previous_interactions = request.previous_interactions
        rag_data = request.rag_data

        # Convert previous interactions into a string
        previous_interactions_str = "\n".join([f"Query: {interaction.query}, Response: {interaction.response}" for interaction in previous_interactions])

        # Process the contextual search using the search agent
        response = search_rec_pb2.ContextualSearchResponse()
        try:
            answer = self.search_agent.process_contextual_query(input_query, previous_interactions_str, rag_data)
            response.answer = answer
        except Exception as e:
            _LOGGER.error(f"SearchAgent failed: {e}")
        return response

# Add this new class for rate limiting
class RateLimitInterceptor(ServerInterceptor):
    def __init__(self, requests_per_minute=60, cleanup_interval=3600):
        self.requests_per_minute = requests_per_minute
        self.request_count = defaultdict(lambda: {'count': 0, 'last_request': 0})
        self.last_cleanup_time = time.time()
        self.cleanup_interval = cleanup_interval  # Cleanup every hour by default

    def intercept(self, method, request, context, method_name):
        current_time = time.time()
        
        # Perform cleanup if necessary
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
            self.last_cleanup_time = current_time

        peer = context.peer()
        if peer.startswith('ipv6:'):
            client_ip = peer.split('%5B')[1].split('%5D')[0]
        else:
            client_ip = peer.split(":")[1]

        client_data = self.request_count[client_ip]
        time_passed = current_time - client_data['last_request']
        
        if time_passed >= 60:
            client_data['count'] = 1
        else:
            client_data['count'] += 1

        client_data['last_request'] = current_time

        if client_data['count'] > self.requests_per_minute:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Rate limit exceeded")

        return method(request, context)

    def _cleanup_old_entries(self, current_time):
        for ip in list(self.request_count.keys()):
            if current_time - self.request_count[ip]['last_request'] > 3600:  # Remove entries older than 1 hour
                del self.request_count[ip]

def _wait_forever(server):
    try:
        while True:
            time.sleep(_ONE_DAY)
    except KeyboardInterrupt:
        print("Stopping the server")
        server.stop(None)

def _run_server():
    _LOGGER.info("Starting new server.")
    options = (("grpc.so_reuseport", 1),)

    # Create the rate limit interceptor
    rate_limit_interceptor = RateLimitInterceptor(requests_per_minute=5)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
        options=options,
        interceptors=(rate_limit_interceptor,)  # Add the interceptor here
    )
    search_rec_pb2_grpc.add_SearchServiceServicer_to_server(
        SearchServicer(), server
    )
    SERVICE_NAMES = (
        search_rec_pb2.DESCRIPTOR.services_by_name['SearchService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port(_BIND_ADDRESS)
    server.start()
    _LOGGER.info(f"Server started on {_BIND_ADDRESS}")
    _LOGGER.info(f"Listening at {_BIND_ADDRESS}")
    _wait_forever(server)

def main():
    multiprocessing.set_start_method("spawn", force=True)
    _LOGGER.info(f"Binding to '{_BIND_ADDRESS}'")
    sys.stdout.flush()
    
    workers = []
    for _ in range(_PROCESS_COUNT):
        worker = multiprocessing.Process(target=_run_server)
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[PID %(process)d] %(message)s")
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    main()
