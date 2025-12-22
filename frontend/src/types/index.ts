// Message input for cases
export interface MessageInput {
  role: 'user' | 'assistant' | 'system'
  message: string
  attachments?: Array<{ file: string }>
  created_at?: string
}

// Expected source reference
export interface ExpectedSource {
  file: string
  page: number
  excerpt?: string
}

// Eval case
export interface Case {
  id: string
  name?: string
  inputs: MessageInput[]
  expected_outcome: string
  expected_source?: ExpectedSource
  created_at: string
  updated_at: string
  dataset_name?: string
  dataset_id?: string
}

// Dataset
export interface Dataset {
  id: string
  name: string
  description?: string
  cases: Case[]
  created_at: string
  updated_at: string
  case_count: number
}

// Eval result
export interface EvalResult {
  case_id: string
  passed: boolean
  actual_output?: string
  judgment?: string
  reasoning?: string
  error_message?: string
  execution_time_ms?: number
  created_at?: string
  case?: Case
  inputs?: MessageInput[]
  expected_outcome?: string
}

// Alias for backward compatibility
export type RunResult = EvalResult

// Run summary (for list view)
export interface RunSummary {
  dataset_id: string
  dataset_name?: string
  filename: string
  eval_name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  passed: number
  failed: number
  total: number
  score: number | null
  started_at: string
  completed_at?: string
}

// Full run with results
export interface Run extends RunSummary {
  results: Record<string, EvalResult>
}

// Baseline
export interface Baseline {
  dataset_id: string
  dataset_name?: string
  saved_at: string
  source_run: string
  passed: number
  failed: number
  total: number
  score: number | null
  results: Record<string, EvalResult>
}

// Settings
export interface Settings {
  model: string
  [key: string]: string
}

// API request types
export interface CreateDatasetRequest {
  name: string
  description?: string
}

export interface CreateCaseRequest {
  dataset_name: string  // Can be name or UUID (server accepts both)
  name?: string
  inputs: MessageInput[]
  expected_outcome: string
  expected_source?: ExpectedSource
}

export interface UpdateCaseRequest {
  name?: string
  inputs?: MessageInput[]
  expected_outcome?: string
  expected_source?: ExpectedSource
}
