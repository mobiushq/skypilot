"""The fake's control plane, for SDK calls made on a sandbox.

A fake sandbox can neither see the host's state nor signal its processes, as a
Modal container reaches Modal only through its API. So the test driver serves
the calls that autodown makes on the sandbox over HTTP on the host's loopback,
which the sandbox shares. The SDK on the sandbox sends the token it found, and
the server checks it, as Modal does.

    FAKE_MODAL_SERVER: the server's URL, in the sandbox's environment.
"""
import http.server
import json
import os
import threading
from typing import Any, Callable, Dict, List, Tuple
import urllib.request

from . import config
from . import exception

ENV_SERVER = 'FAKE_MODAL_SERVER'


def is_client() -> bool:
    """On a sandbox: the server's URL is set and the host's state is not."""
    return ENV_SERVER in os.environ and 'FAKE_MODAL_STATE' not in os.environ


def call(op: str, **params: Any) -> Any:
    body = json.dumps({
        'op': op,
        'token': [
            config.config.get('token_id'),
            config.config.get('token_secret')
        ],
        'params': params,
    }).encode()
    request = urllib.request.Request(
        os.environ[ENV_SERVER],
        data=body,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.load(response)
    if 'error' in reply:
        raise getattr(exception, reply['error'])(reply['message'])
    return reply['result']


def serve(
    handlers: Dict[str, Callable[...,
                                 Any]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Serves handlers[op](**params) in a thread.

    Returns the URL, and the list that each call's op and outcome is added to.
    """
    token = [
        os.environ.get('MODAL_TOKEN_ID'),
        os.environ.get('MODAL_TOKEN_SECRET')
    ]
    served: List[Tuple[str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_POST(self):  # pylint: disable=invalid-name
            length = int(self.headers['Content-Length'])
            request = json.loads(self.rfile.read(length))
            try:
                if request['token'] != token:
                    raise exception.AuthError('Token missing or invalid.')
                reply = {'result': handlers[request['op']](**request['params'])}
                served.append((request['op'], 'ok'))
            except exception.Error as e:
                reply = {'error': type(e).__name__, 'message': str(e)}
                served.append((request['op'], type(e).__name__))
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # pylint: disable=arguments-differ
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{server.server_port}', served
