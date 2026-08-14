"""Media generation services (speech, image, avatar video).

The gRPC c-ares DNS resolver fails inside some container networks, which shows up
as "DNS resolution failed" from the Google clients. The native resolver has to be
selected before grpc creates a channel, so it is set here — this package is
imported before any media client is constructed.
"""

import os

os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
