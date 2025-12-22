import { useState, useEffect } from 'react'
import { useStore } from '@/hooks/useStore'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const AVAILABLE_MODELS = [
  { id: 'anthropic/claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5', provider: 'Anthropic' },
  { id: 'anthropic/claude-sonnet-4-20250514', name: 'Claude Sonnet 4', provider: 'Anthropic' },
  { id: 'anthropic/claude-opus-4-20250514', name: 'Claude Opus 4', provider: 'Anthropic' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
  { id: 'gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  { id: 'gemini/gemini-2.0-flash', name: 'Gemini 2.0 Flash', provider: 'Google' },
]

export function Settings() {
  const { state, updateSettings, loadSettings } = useStore()
  const { settings, loading } = state
  const [model, setModel] = useState(AVAILABLE_MODELS[0].id)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!settings) {
      loadSettings()
    }
  }, [settings, loadSettings])

  useEffect(() => {
    if (settings?.model) {
      setModel(settings.model)
    }
  }, [settings])

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateSettings({ model })
    } finally {
      setSaving(false)
    }
  }

  if (loading.settings) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <div className="mr-2 h-4 w-4 border-2 border-zinc-200 border-t-taro rounded-full animate-spin" />
        Loading settings...
      </div>
    )
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
      </div>

      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle className="text-base">LLM Configuration</CardTitle>
          <p className="text-sm text-zinc-500">
            Configure the model used for generating test cases and judging results.
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label htmlFor="model" className="block text-sm font-medium text-zinc-600 mb-1.5">
                Model
              </label>
              <select
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full max-w-sm px-3 py-2 border border-zinc-200 bg-white text-sm focus:outline-none focus:ring-1 focus:ring-taro"
              >
                {AVAILABLE_MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.provider})
                  </option>
                ))}
              </select>
            </div>

            <Button onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          </div>

          <div className="mt-8 p-4 bg-zinc-100">
            <h4 className="text-sm font-semibold mb-2">API Keys</h4>
            <p className="text-xs text-zinc-500 mb-3">
              Set the appropriate environment variable for your provider before starting simboba:
            </p>
            <pre className="p-3 bg-zinc-900 text-zinc-100 text-xs overflow-x-auto font-mono">
{`export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...`}
            </pre>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
