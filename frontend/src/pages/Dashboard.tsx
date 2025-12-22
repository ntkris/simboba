import { Link } from 'react-router-dom'
import { useStore } from '@/hooks/useStore'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { cn, relativeTime } from '@/lib/utils'

export function Dashboard() {
  const { state } = useStore()
  const { datasets, runs, loading } = state

  if (loading.datasets || loading.runs) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <div className="mr-2 h-4 w-4 border-2 border-zinc-200 border-t-taro rounded-full animate-spin" />
        Loading...
      </div>
    )
  }

  if (datasets.length === 0) {
    return (
      <Card>
        <CardContent className="py-20 text-center">
          <h3 className="text-base font-semibold mb-2">Get started</h3>
          <p className="text-zinc-500 mb-5">Create your first dataset to start running evaluations.</p>
          <Button asChild>
            <Link to="/datasets">Create Dataset</Link>
          </Button>
        </CardContent>
      </Card>
    )
  }

  if (runs.length === 0) {
    return (
      <Card>
        <CardContent className="py-20 text-center">
          <h3 className="text-base font-semibold mb-2">Ready to run</h3>
          <p className="text-zinc-500 mb-5">
            You have {datasets.length} dataset{datasets.length > 1 ? 's' : ''}. Run your first evaluation.
          </p>
          <Button asChild>
            <Link to="/datasets">Go to Datasets</Link>
          </Button>
        </CardContent>
      </Card>
    )
  }

  const recentRuns = runs.slice(0, 10)

  return (
    <div>
      <div className="flex justify-between items-start mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Recent Runs</h1>
      </div>

      <Card>
        <CardContent className="p-0">
          {recentRuns.map((run) => {
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
            const casesLabel = run.total === 1 && !run.dataset_id ? '1 eval' : `${run.total} cases`

            return (
              <Link
                key={`${run.dataset_id}-${run.filename}`}
                to={`/runs/${run.dataset_id || '_adhoc'}/${run.filename}`}
                className="flex items-center px-4 py-3.5 border-b border-zinc-100 last:border-b-0 hover:bg-zinc-50 transition-colors"
              >
                <div className="flex-1">
                  <div className="font-medium">
                    {run.status === 'running' && (
                      <span className="inline-block w-1.5 h-1.5 bg-taro rounded-full mr-2 animate-pulse" />
                    )}
                    {title}
                    {run.status === 'running' && <span className="text-taro ml-1">Running</span>}
                  </div>
                  <div className="text-xs text-zinc-500 mt-0.5">
                    {relativeTime(run.started_at)} · {casesLabel}
                    {dataset && ` · ${run.eval_name}`}
                  </div>
                </div>
                <span className={cn('font-mono text-sm font-medium', rateClass)}>
                  {run.status === 'running'
                    ? '—'
                    : run.score !== null
                      ? `${run.score.toFixed(0)}%`
                      : '—'}
                </span>
                <span className="text-zinc-300 ml-2">›</span>
              </Link>
            )
          })}
        </CardContent>
      </Card>

      <p className="mt-4">
        <Link to="/runs" className="text-sm text-zinc-500 hover:text-zinc-900">
          View all runs →
        </Link>
      </p>
    </div>
  )
}
