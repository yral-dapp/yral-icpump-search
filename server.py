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

from search_agent_bq import SearchAgent

_LOGGER = logging.getLogger(__name__)

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

class SignatureValidationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        def abort(ignored_request, context):
            _LOGGER.warning("Aborting request due to invalid signature")
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)

    def intercept_service(self, continuation, handler_call_details):
        metadata_dict = dict(handler_call_details.invocation_metadata)
        try:
            token = metadata_dict[_AUTH_HEADER_KEY].split()[1]
            payload = jwt.decode(
                token,
                _PUBLIC_KEY,
                algorithms=["EdDSA"],
            )

            if payload == _JWT_PAYLOAD:
                return continuation(handler_call_details)
            else:
                _LOGGER.warning(f"Received invalid payload: {payload}")
                return self._abort_handler
        except Exception as e:
            _LOGGER.error(f"Exception occurred during token validation: {e}")
            return self._abort_handler
class SearchServicer(search_rec_pb2_grpc.SearchServiceServicer):
    def __init__(self):
        self.search_agent = SearchAgent() 

    def Search(self, request, context):
        search_query = request.input_query
        _LOGGER.info(f"Received search query: {search_query}")
        df, answer = self.search_agent.process_query(search_query)
        response = search_rec_pb2.SearchResponse()
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
        return response

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

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
        interceptors=(SignatureValidationInterceptor(),),
        options=options
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