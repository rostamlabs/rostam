// SPDX-License-Identifier: Apache-2.0

package vector

import "github.com/rostamlabs/rostam/sdk/vtypes"

// This file re-exports the engine-free data types that now live in the vtypes
// leaf package, so the engine and every existing caller compile unchanged while
// the wire codec and network client can depend on vtypes without pulling in the
// engine. See package vtypes.

// Moved types.
type (
	ValueKind         = vtypes.ValueKind
	Value             = vtypes.Value
	Metadata          = vtypes.Metadata
	SparseVector      = vtypes.SparseVector
	Metric            = vtypes.Metric
	IndexType         = vtypes.IndexType
	QuantMode         = vtypes.QuantMode
	FullTextConfig    = vtypes.FullTextConfig
	Config            = vtypes.Config
	NamedVectorParams = vtypes.NamedVectorParams
	QuantStore        = vtypes.QuantStore
	FilterOp          = vtypes.FilterOp
	Filter            = vtypes.Filter
	GeoCondition      = vtypes.GeoCondition

	// Query / result / ordering / record / group / multivector clusters.
	LeafKind          = vtypes.LeafKind
	QueryMode         = vtypes.QueryMode
	QueryLeaf         = vtypes.QueryLeaf
	QuerySource       = vtypes.QuerySource
	QuerySpec         = vtypes.QuerySpec
	QueryResult       = vtypes.QueryResult
	RecommendStrategy = vtypes.RecommendStrategy
	ContextPair       = vtypes.ContextPair
	DiscoverPair      = vtypes.DiscoverPair
	FusionMethod      = vtypes.FusionMethod
	Result            = vtypes.Result
	HybridOpts        = vtypes.HybridOpts
	GroupOpts         = vtypes.GroupOpts
	Group             = vtypes.Group
	Document          = vtypes.Document
	RawDocument       = vtypes.RawDocument
	RawGroup          = vtypes.RawGroup
	OrderKind         = vtypes.OrderKind
	OrderBy           = vtypes.OrderBy
	OrderVal          = vtypes.OrderVal
	MultiVectorConfig = vtypes.MultiVectorConfig
	MultiResult       = vtypes.MultiResult
	MultiScanRecord   = vtypes.MultiScanRecord
	ScanRecord        = vtypes.ScanRecord
	BM25GlobalStats   = vtypes.BM25GlobalStats
)

// Moved constants (values preserved verbatim, so wire/snapshot encodings are
// byte-identical).
const (
	ValueNone    = vtypes.ValueNone
	ValueString  = vtypes.ValueString
	ValueInt     = vtypes.ValueInt
	ValueFloat   = vtypes.ValueFloat
	ValueBool    = vtypes.ValueBool
	ValueStrings = vtypes.ValueStrings
	ValueInts    = vtypes.ValueInts
	ValueFloats  = vtypes.ValueFloats
	ValueGeo     = vtypes.ValueGeo
	ValueRecord  = vtypes.ValueRecord

	Cosine     = vtypes.Cosine
	L2         = vtypes.L2
	DotProduct = vtypes.DotProduct

	IndexHNSW   = vtypes.IndexHNSW
	IndexIVF    = vtypes.IndexIVF
	IndexVamana = vtypes.IndexVamana
	IndexGPU    = vtypes.IndexGPU

	QuantNone = vtypes.QuantNone
	QuantSQ8  = vtypes.QuantSQ8
	QuantBQ1  = vtypes.QuantBQ1
	QuantPQ   = vtypes.QuantPQ
	QuantSQ   = vtypes.QuantSQ
	QuantPRQ  = vtypes.QuantPRQ

	QuantInRAM = vtypes.QuantInRAM
	QuantMmap  = vtypes.QuantMmap

	FilterAnd        = vtypes.FilterAnd
	FilterOr         = vtypes.FilterOr
	FilterNot        = vtypes.FilterNot
	FilterEq         = vtypes.FilterEq
	FilterNe         = vtypes.FilterNe
	FilterGt         = vtypes.FilterGt
	FilterGte        = vtypes.FilterGte
	FilterLt         = vtypes.FilterLt
	FilterLte        = vtypes.FilterLte
	FilterIn         = vtypes.FilterIn
	FilterContains   = vtypes.FilterContains
	FilterMatch      = vtypes.FilterMatch
	FilterRegex      = vtypes.FilterRegex
	FilterIsEmpty    = vtypes.FilterIsEmpty
	FilterIsNull     = vtypes.FilterIsNull
	FilterDtGt       = vtypes.FilterDtGt
	FilterDtGte      = vtypes.FilterDtGte
	FilterDtLt       = vtypes.FilterDtLt
	FilterDtLte      = vtypes.FilterDtLte
	FilterGeoRadius  = vtypes.FilterGeoRadius
	FilterGeoBox     = vtypes.FilterGeoBox
	FilterGeoPolygon = vtypes.FilterGeoPolygon
	FilterRowExists  = vtypes.FilterRowExists
	FilterRowAbsent  = vtypes.FilterRowAbsent

	// Query leaf kinds.
	LeafDense     = vtypes.LeafDense
	LeafSparse    = vtypes.LeafSparse
	LeafMVMaxSim  = vtypes.LeafMVMaxSim
	LeafRecommend = vtypes.LeafRecommend
	LeafDiscover  = vtypes.LeafDiscover

	// Query combine modes.
	ModeFusion = vtypes.ModeFusion
	ModeRerank = vtypes.ModeRerank

	// Recommend strategies.
	RecommendAverageVector = vtypes.RecommendAverageVector
	RecommendBestScore     = vtypes.RecommendBestScore

	// Fusion methods.
	FusionRRF      = vtypes.FusionRRF
	FusionWeighted = vtypes.FusionWeighted
	FusionDBSF     = vtypes.FusionDBSF

	// Order-by value kinds.
	OrderNumeric  = vtypes.OrderNumeric
	OrderDatetime = vtypes.OrderDatetime
	OrderString   = vtypes.OrderString

	// Query spec breadth bound (shared with the ops decode).
	MaxPrefetchSources = vtypes.MaxPrefetchSources
)

// Moved error sentinels (value aliases preserve errors.Is identity) and the
// Value constructors (re-exported so every existing call site is unchanged).
var (
	ErrSparseMismatch = vtypes.ErrSparseMismatch
	ErrSparseUnsorted = vtypes.ErrSparseUnsorted

	// Moved constructors / helpers (re-exported as function values so every
	// existing call site is unchanged).
	LeafSource  = vtypes.LeafSource
	WithContent = vtypes.WithContent

	// Moved error sentinels (value aliases preserve errors.Is identity).
	ErrDimMismatch            = vtypes.ErrDimMismatch
	ErrQuerySpecTooDeep       = vtypes.ErrQuerySpecTooDeep
	ErrTooManyPrefetchSources = vtypes.ErrTooManyPrefetchSources

	NewString  = vtypes.NewString
	NewInt     = vtypes.NewInt
	NewFloat   = vtypes.NewFloat
	NewBool    = vtypes.NewBool
	NewStrings = vtypes.NewStrings
	NewInts    = vtypes.NewInts
	NewFloats  = vtypes.NewFloats
	NewGeo     = vtypes.NewGeo
	NewRecord  = vtypes.NewRecord

	// DefaultConfig is re-exported as a function value so every DefaultConfig()
	// call site is unchanged.
	DefaultConfig = vtypes.DefaultConfig

	// Config validation error sentinels (returned by ValidateConfig) now live in
	// vtypes; value aliases preserve errors.Is identity.
	ErrInvalidDim                   = vtypes.ErrInvalidDim
	ErrInvalidMetric                = vtypes.ErrInvalidMetric
	ErrInvalidM                     = vtypes.ErrInvalidM
	ErrInvalidEf                    = vtypes.ErrInvalidEf
	ErrInvalidQuant                 = vtypes.ErrInvalidQuant
	ErrInvalidRescoreFactor         = vtypes.ErrInvalidRescoreFactor
	ErrInvalidQuantStorage          = vtypes.ErrInvalidQuantStorage
	ErrInvalidGraphMmap             = vtypes.ErrInvalidGraphMmap
	ErrInvalidPersistent            = vtypes.ErrInvalidPersistent
	ErrInvalidQuantizedBuild        = vtypes.ErrInvalidQuantizedBuild
	ErrInvalidWAL                   = vtypes.ErrInvalidWAL
	ErrInvalidPartitions            = vtypes.ErrInvalidPartitions
	ErrInvalidIndexType             = vtypes.ErrInvalidIndexType
	ErrGPUNotCompiled               = vtypes.ErrGPUNotCompiled
	ErrInvalidIVFParams             = vtypes.ErrInvalidIVFParams
	ErrInvalidVamanaParams          = vtypes.ErrInvalidVamanaParams
	ErrInvalidIVFTrainThreshold     = vtypes.ErrInvalidIVFTrainThreshold
	ErrInvalidFilterFirstRelativeBP = vtypes.ErrInvalidFilterFirstRelativeBP
	ErrInvalidIVFDriftFactor        = vtypes.ErrInvalidIVFDriftFactor
	ErrInvalidIVFPQ                 = vtypes.ErrInvalidIVFPQ
	ErrInvalidIVFPQM                = vtypes.ErrInvalidIVFPQM
	ErrInvalidQuantPQM              = vtypes.ErrInvalidQuantPQM
	ErrInvalidPQNBits               = vtypes.ErrInvalidPQNBits
	ErrInvalidPRQLayers             = vtypes.ErrInvalidPRQLayers
	ErrInvalidOPQ                   = vtypes.ErrInvalidOPQ
	ErrInvalidOPQIters              = vtypes.ErrInvalidOPQIters
	ErrInvalidPQDropVecs            = vtypes.ErrInvalidPQDropVecs
	ErrInvalidAnisotropicEta        = vtypes.ErrInvalidAnisotropicEta
	ErrInvalidSOAR                  = vtypes.ErrInvalidSOAR
	ErrInvalidSOARLambda            = vtypes.ErrInvalidSOARLambda
	ErrInvalidFullText              = vtypes.ErrInvalidFullText
)
