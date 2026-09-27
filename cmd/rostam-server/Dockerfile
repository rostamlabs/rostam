# Multi-stage build for rostam-server (the REST/gRPC/TCP front door).
# Build context must be the repo root:  docker build -f cmd/rostam-server/Dockerfile .
#
# cgo is required (wasmtime-go is pulled in transitively), so the build image
# carries a C toolchain and the runtime image is debian-slim (not scratch).
#
# Dashboard UI: built in its own Node stage and copied into dashboard/dist/
# (the go:embed'd tree) before the Go build, so the shipped binary embeds the
# real Vite output instead of the checked-in .gitkeep placeholder.
FROM node:20-bookworm AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum go.work go.work.sum ./
COPY sdk/go.mod sdk/go.sum ./sdk/
COPY client/go.mod client/go.sum ./client/
RUN go mod download
COPY . .
# Overwrites the placeholder dashboard/dist/ (just .gitkeep in a checkout) with
# the real built assets from the ui stage above.
COPY --from=ui /ui/dist/. ./dashboard/dist/
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/rostam-server ./cmd/rostam-server

FROM debian:bookworm-slim
# Proves to the official MCP Registry that this image is the one server.json
# lists (`docker run -i --rm <image> mcp` is the stdio server).
LABEL io.modelcontextprotocol.server.name="io.github.rostamlabs/rostam"
RUN useradd -m -u 10001 rostam && mkdir -p /data && chown rostam:rostam /data
COPY --from=build /out/rostam-server /usr/local/bin/rostam-server
USER rostam
# Writable working dir: the vector store creates a "vectors/" dir relative to
# the data dir (empty data dir → relative to CWD), so CWD must be writable by
# the rostam user even in the in-memory default.
WORKDIR /data
# 8080 = REST, 7000 = binary TCP. Both are only DOCUMENTED here; neither is
# reachable until you publish it with -p. TCP is bound by default because the Go
# remote client (rostam.NewClient) speaks only the binary protocol — without it,
# a Go service could not reach the container without overriding the command.
EXPOSE 8080 7000
# Single-node, in-memory by default (fast HNSW, no disk). Override the flags to
# enable persistence / mmap (-persistent-vectors -cluster ... -data /data) or to
# add gRPC (-grpc :9090).
#
# AUTHENTICATION IS REQUIRED. The default CMD binds 0.0.0.0, which the server
# refuses to serve unauthenticated, so pass a key:
#
#     docker run -p 8080:8080 -e ROSTAM_API_KEY=<token> rostam-server:latest   # REST only
#     docker run -p 8080:8080 -p 7000:7000 -e ROSTAM_API_KEY=<token> ...        # + Go/TCP clients
#
# The token guards every transport identically, so publishing 7000 is no less
# safe than 8080. Publish only the ports you use: an unpublished listener is not
# reachable from the host, though other containers on the same Docker network
# can still reach it on the container's port (and host networking publishes
# everything) — which is why the auth gate, not port publication, is the control.
#
# ROSTAM_API_KEY is read instead of -api-key so the secret never appears in the
# process table or in `docker inspect`'s Cmd. For a deliberately open dev
# container, append -insecure to the command instead.
ENTRYPOINT ["rostam-server"]
CMD ["-http", "0.0.0.0:8080", "-tcp", "0.0.0.0:7000"]
