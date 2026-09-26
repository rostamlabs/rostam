module github.com/rostamlabs/rostam/client

go 1.26.5

require (
	github.com/cespare/xxhash/v2 v2.3.0
	github.com/jackc/puddle/v2 v2.2.2
	github.com/rostamlabs/rostam/sdk v0.2.0
)

require (
	golang.org/x/sync v0.1.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)

replace github.com/rostamlabs/rostam/sdk => ../sdk
