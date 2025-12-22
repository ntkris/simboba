import { useState, useEffect, useCallback } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { User, Bot } from 'lucide-react'
import { useStore } from '@/hooks/useStore'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn, relativeTime, formatDuration } from '@/lib/utils'
import * as api from '@/lib/api'
import type { Run, RunResult, MessageInput, Case } from '@/types'

export function Runs() {
  const { datasetId, filename } = useParams()
  const navigate = useNavigate()
  const { state, deleteRun, showToast } = useStore()
  const { datasets, runs, loading } = state

  const [selectedRun, setSelectedRun] = useState<Run | null>(null)
  const [loadingRun, setLoadingRun] = useState(false)

  const loadRunDetail = useCallback(async (dsId: string, fname: string) => {
    setLoadingRun(true)
    try {
      const run = await api.getRun(dsId, fname)
      setSelectedRun(run)
    } catch (e) {
      showToast('Failed to load run details', true)
      navigate('/runs')
    } finally {
      setLoadingRun(false)
    }
  }, [navigate, showToast])

  useEffect(() => {
    if (datasetId && filename) {
      loadRunDetail(datasetId, filename)
    } else {
      setSelectedRun(null)
    }
  }, [datasetId, filename, loadRunDetail])

  const handleDeleteRun = async () => {
    if (!selectedRun) return
    if (!confirm('Delete this run?')) return

    try {
      await deleteRun(selectedRun.dataset_id || '_adhoc', selectedRun.filename)
      setSelectedRun(null)
      navigate('/runs')
    } catch {
      showToast('Failed to delete run', true)
    }
  }

  const closeRunSidebar = () => {
    setSelectedRun(null)
    navigate('/runs')
  }

  if (loading.runs) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <div className="mr-2 h-4 w-4 border-2 border-zinc-200 border-t-taro rounded-full animate-spin" />
        Loading...
      </div>
    )
  }

  return (
    <>
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Runs</h1>
          <p className="text-sm text-zinc-500 mt-1">All evaluation runs</p>
        </div>
      </div>

      {runs.length === 0 ? (
        <Card>
          <CardContent className="py-20 text-center">
            <h3 className="text-base font-semibold mb-2">No runs yet</h3>
            <p className="text-zinc-500 mb-5">Run your first evaluation to see results here.</p>
            <Button asChild>
              <Link to="/datasets">Create Dataset</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            {runs.map((run) => {
              const dataset = datasets.find((d) => d.id === run.dataset_id)
              const title = dataset?.name || run.dataset_name || run.eval_name || 'Single Eval'
              const rateClass =
                run.score !== null
                  ? run.score >= 80
                    ? 'text-green-500'
                    : run.score >= 60
                      ? 'text-zinc-600'
                      : 'text-red-500'
                  : ''
              const isRunning = run.status === 'running'
              const casesLabel = run.total === 1 && !run.dataset_id ? '1 eval' : `${run.total} cases`
              const dsId = run.dataset_id || '_adhoc'
              const isSelected = selectedRun?.filename === run.filename && (selectedRun?.dataset_id || '_adhoc') === dsId

              return (
                <div
                  key={`${dsId}-${run.filename}`}
                  onClick={() => navigate(`/runs/${dsId}/${run.filename}`)}
                  className={`flex items-center px-4 py-3.5 border-b border-zinc-100 last:border-b-0 cursor-pointer transition-colors ${
                    isSelected ? 'bg-taro/10' : 'hover:bg-zinc-50'
                  }`}
                >
                  <div className="flex-1">
                    <div className="font-medium">
                      {isRunning && (
                        <span className="inline-block w-1.5 h-1.5 bg-taro rounded-full mr-2 animate-pulse" />
                      )}
                      {title}
                      {isRunning && <span className="text-taro ml-1">Running</span>}
                    </div>
                    <div className="text-xs text-zinc-500 mt-0.5">
                      {relativeTime(run.started_at)} · {casesLabel}
                      {dataset && ` · ${run.eval_name}`}
                    </div>
                  </div>
                  <span className={cn('font-mono text-sm font-medium', rateClass)}>
                    {isRunning ? '—' : run.score !== null ? `${run.score.toFixed(0)}%` : '—'}
                  </span>
                  <span className="text-zinc-300 ml-2">›</span>
                </div>
              )
            })}
          </CardContent>
        </Card>
      )}

      {/* Run sidebar */}
      <RunSidebar
        run={selectedRun}
        loading={loadingRun}
        datasets={datasets}
        onClose={closeRunSidebar}
        onDelete={handleDeleteRun}
      />
    </>
  )
}

function RunSidebar({
  run,
  loading,
  datasets,
  onClose,
  onDelete,
}: {
  run: Run | null
  loading: boolean
  datasets: { id: string; name: string }[]
  onClose: () => void
  onDelete: () => void
}) {
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set())

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && run) {
        onClose()
      }
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [run, onClose])

  useEffect(() => {
    setExpandedResults(new Set())
  }, [run])

  if (!run && !loading) return null

  const dataset = run?.dataset_id ? datasets.find((d) => d.id === run.dataset_id) : null
  const title = dataset?.name || run?.dataset_name || run?.eval_name || 'Single Eval'

  // Convert results dict to array
  const resultsObj = run?.results || {}
  const results: (RunResult & { case_id: string })[] = Object.entries(resultsObj).map(
    ([caseId, result]) => ({
      ...(result as RunResult),
      case_id: caseId,
    })
  )

  const duration =
    run?.completed_at && run?.started_at
      ? formatDuration(new Date(run.completed_at).getTime() - new Date(run.started_at).getTime())
      : 'In progress'

  const toggleResult = (id: string) => {
    setExpandedResults((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/30 z-40" onClick={onClose} />

      {/* Sidebar */}
      <div className="fixed top-0 right-0 w-[560px] h-full bg-white border-l border-zinc-200 z-50 flex flex-col">
        <div className="flex justify-between items-center px-5 py-4 border-b border-zinc-200 bg-zinc-50">
          <span className="font-semibold">{loading ? 'Loading...' : title}</span>
          <button onClick={onClose} className="text-2xl text-zinc-400 hover:text-zinc-900">
            ×
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
              <div className="mr-2 h-4 w-4 border-2 border-zinc-200 border-t-taro rounded-full animate-spin" />
              Loading run details...
            </div>
          ) : run ? (
            <>
              <div className="mb-6">
                <div className="text-sm text-zinc-500 mb-4">
                  <span>{relativeTime(run.started_at)}</span>
                  <span className="mx-2">·</span>
                  <span>{duration}</span>
                  <span className="mx-2">·</span>
                  <span>{run.eval_name}</span>
                </div>

                <div className="grid grid-cols-4 gap-3">
                  <div className="text-center p-3 bg-zinc-50">
                    <div className="font-mono text-xl font-medium">{run.total}</div>
                    <div className="text-xs uppercase tracking-wide text-zinc-400 mt-1">Total</div>
                  </div>
                  <div className="text-center p-3 bg-zinc-50">
                    <div className="font-mono text-xl font-medium text-green-500">{run.passed}</div>
                    <div className="text-xs uppercase tracking-wide text-zinc-400 mt-1">Passed</div>
                  </div>
                  <div className="text-center p-3 bg-zinc-50">
                    <div className="font-mono text-xl font-medium text-red-500">{run.failed}</div>
                    <div className="text-xs uppercase tracking-wide text-zinc-400 mt-1">Failed</div>
                  </div>
                  <div className="text-center p-3 bg-zinc-50">
                    <div className="font-mono text-xl font-medium text-taro">
                      {run.score !== null ? `${run.score.toFixed(0)}%` : '—'}
                    </div>
                    <div className="text-xs uppercase tracking-wide text-zinc-400 mt-1">Score</div>
                  </div>
                </div>
              </div>

              <div>
                <div className="text-xs font-medium uppercase tracking-wide text-zinc-400 mb-3">
                  Results
                </div>
                {results.map((res) => (
                  <ResultItem
                    key={res.case_id}
                    result={res}
                    expanded={expandedResults.has(res.case_id)}
                    onToggle={() => toggleResult(res.case_id)}
                  />
                ))}
              </div>
            </>
          ) : null}
        </div>

        <div className="px-5 py-4 border-t border-zinc-200 flex gap-2 justify-end">
          <Button
            variant="ghost"
            size="sm"
            className="text-red-500 hover:text-red-600 hover:bg-red-50"
            onClick={onDelete}
          >
            Delete Run
          </Button>
        </div>
      </div>
    </>
  )
}

function ResultItem({
  result,
  expanded,
  onToggle,
}: {
  result: RunResult & { case_id: string }
  expanded: boolean
  onToggle: () => void
}) {
  const c: Partial<Case> = result.case || {}
  const inputs: MessageInput[] = c.inputs || result.inputs || []

  // Determine name
  let name: string
  if (result.case_id && result.case_id !== '_adhoc') {
    name = c.name || `Case #${result.case_id.substring(0, 8)}`
  } else {
    const firstMessage = inputs[0]?.message || ''
    name = firstMessage.length > 30 ? firstMessage.substring(0, 30) + '...' : firstMessage || 'Eval'
  }

  const expectedOutcome = c.expected_outcome || result.expected_outcome || '—'

  return (
    <div className="border border-zinc-200 mb-3 overflow-hidden">
      <div
        onClick={onToggle}
        className="flex items-center px-3.5 py-3 cursor-pointer hover:bg-zinc-50 transition-colors"
      >
        <span
          className={cn(
            'text-zinc-400 mr-2.5 transition-transform',
            expanded && 'rotate-90'
          )}
        >
          ›
        </span>
        <div className="flex-1">
          <span className="font-medium text-sm">{name}</span>
        </div>
        <Badge variant={result.passed ? 'success' : 'error'}>
          <span
            className={cn(
              'inline-block w-1.5 h-1.5 rounded-full mr-1.5',
              result.passed ? 'bg-green-500' : 'bg-red-500'
            )}
          />
          {result.passed ? 'Pass' : 'Fail'}
        </Badge>
      </div>

      {expanded && (
        <div className="px-3.5 py-3.5 bg-zinc-50 border-t border-zinc-100">
          {/* Conversation + Actual Output */}
          {(inputs.length > 0 || result.actual_output) && (
            <div className="mb-4">
              <div className="text-xs font-medium uppercase tracking-wide text-zinc-400 mb-2">
                Conversation
              </div>
              <div className="space-y-2">
                {inputs.map((m, i) => {
                  const isUser = m.role === 'user'
                  return (
                    <div
                      key={i}
                      className={`flex ${isUser ? 'justify-start' : 'justify-end'}`}
                    >
                      <div
                        className={`flex items-start gap-2 max-w-[85%] ${
                          isUser ? 'flex-row' : 'flex-row-reverse'
                        }`}
                      >
                        <div
                          className={`shrink-0 w-6 h-6 flex items-center justify-center ${
                            isUser ? 'bg-taro/10 text-taro' : 'bg-zinc-200 text-zinc-600'
                          }`}
                        >
                          {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
                        </div>
                        <div
                          className={`px-2.5 py-1.5 text-sm leading-relaxed ${
                            isUser ? 'bg-taro/10 text-zinc-900' : 'bg-zinc-200 text-zinc-900'
                          }`}
                        >
                          {m.message}
                        </div>
                      </div>
                    </div>
                  )
                })}
                {/* Actual Output as final agent message */}
                {result.actual_output && (
                  <div className="flex justify-end">
                    <div className="flex items-start gap-2 max-w-[85%] flex-row-reverse">
                      <div className="shrink-0 w-6 h-6 flex items-center justify-center bg-zinc-200 text-zinc-600">
                        <Bot className="w-3.5 h-3.5" />
                      </div>
                      <div
                        className={cn(
                          'px-2.5 py-1.5 text-sm leading-relaxed bg-zinc-200 text-zinc-900 border',
                          result.passed ? 'border-green-400' : 'border-red-400'
                        )}
                      >
                        {result.actual_output}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Expected Outcome */}
          <div className="mb-4">
            <div className="text-xs font-medium uppercase tracking-wide text-zinc-400 mb-2">
              Expected Outcome
            </div>
            <div className="p-3 bg-white border border-zinc-200 text-sm">{expectedOutcome}</div>
          </div>

          {/* Judge Reasoning */}
          {result.reasoning && (
            <div className="mb-4">
              <div className="text-xs font-medium uppercase tracking-wide text-zinc-400 mb-2">
                Judge Reasoning
              </div>
              <div className="p-3 bg-zinc-100 text-sm text-zinc-600">{result.reasoning}</div>
            </div>
          )}

          {/* Error */}
          {result.error_message && (
            <div className="p-3 bg-red-50 border border-red-200 text-sm text-red-700">
              <div className="text-xs font-medium uppercase tracking-wide text-red-700 mb-1">
                Error
              </div>
              {result.error_message}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
