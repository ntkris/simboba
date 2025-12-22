import { useState, useEffect, useCallback } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { useStore } from '@/hooks/useStore'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { relativeTime } from '@/lib/utils'
import * as api from '@/lib/api'
import type { Case, Dataset } from '@/types'

export function Datasets() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { state, deleteDataset, showToast, loadDatasets } = useStore()
  const { datasets, loading } = state

  const [currentDataset, setCurrentDataset] = useState<Dataset | null>(null)
  const [cases, setCases] = useState<Case[]>([])
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [selectedCase, setSelectedCase] = useState<Case | null>(null)
  const [showDeleteModal, setShowDeleteModal] = useState<{ id: string; name: string; count: number } | null>(null)
  const [showCaseModal, setShowCaseModal] = useState(false)
  const [editingCase, setEditingCase] = useState<Case | null>(null)
  const [showNewDatasetModal, setShowNewDatasetModal] = useState(false)

  const loadDatasetDetail = useCallback(async (datasetId: string) => {
    setLoadingDetail(true)
    try {
      const [ds, casesData] = await Promise.all([
        api.getDataset(datasetId),
        api.listCases(datasetId),
      ])
      setCurrentDataset(ds)
      setCases(casesData)
    } catch (e) {
      showToast('Failed to load dataset', true)
      navigate('/datasets')
    } finally {
      setLoadingDetail(false)
    }
  }, [navigate, showToast])

  useEffect(() => {
    if (id) {
      loadDatasetDetail(id)
    } else {
      setCurrentDataset(null)
      setCases([])
      setSelectedCase(null)
    }
  }, [id, loadDatasetDetail])

  const handleDelete = async () => {
    if (!showDeleteModal) return
    try {
      await deleteDataset(showDeleteModal.id)
      setShowDeleteModal(null)
      navigate('/datasets')
    } catch {
      showToast('Failed to delete dataset', true)
    }
  }

  const handleDeleteCase = async (caseId: string) => {
    if (!confirm('Delete this case?')) return
    if (!currentDataset) return
    try {
      await api.deleteCase(currentDataset.id, caseId)
      loadDatasetDetail(currentDataset.id)
      setSelectedCase(null)
      showToast('Case deleted')
    } catch {
      showToast('Failed to delete case', true)
    }
  }

  if (loading.datasets) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <div className="mr-2 h-4 w-4 border-2 border-zinc-200 border-t-taro rounded-full animate-spin" />
        Loading...
      </div>
    )
  }

  // Dataset detail view
  if (id && currentDataset) {
    return (
      <>
        <DatasetDetail
          dataset={currentDataset}
          cases={cases}
          loading={loadingDetail}
          onSelectCase={setSelectedCase}
          selectedCaseId={selectedCase?.id}
          onAddCase={() => {
            setEditingCase(null)
            setShowCaseModal(true)
          }}
        />

        {/* Case sidebar */}
        <CaseSidebar
          caseData={selectedCase}
          onClose={() => setSelectedCase(null)}
          onEdit={(c) => {
            setSelectedCase(null)
            setEditingCase(c)
            setShowCaseModal(true)
          }}
          onDelete={(caseId) => handleDeleteCase(caseId)}
        />

        {/* Case modal */}
        <CaseModal
          open={showCaseModal}
          onOpenChange={setShowCaseModal}
          editingCase={editingCase}
          datasetId={currentDataset.id}
          onSaved={() => {
            setShowCaseModal(false)
            loadDatasetDetail(currentDataset.id)
          }}
        />
      </>
    )
  }

  // Datasets list view
  return (
    <>
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
          <p className="text-sm text-zinc-500 mt-1">Manage your evaluation datasets</p>
        </div>
        <Button onClick={() => setShowNewDatasetModal(true)}>+ New Dataset</Button>
      </div>

      {datasets.length === 0 ? (
        <Card>
          <CardContent className="py-20 text-center">
            <h3 className="text-base font-semibold mb-2">No datasets yet</h3>
            <p className="text-zinc-500 mb-5">Create your first dataset to start running evaluations.</p>
            <Button onClick={() => setShowNewDatasetModal(true)}>+ New Dataset</Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            <table className="w-full">
              <thead>
                <tr className="border-b border-zinc-200 bg-zinc-50">
                  <th className="text-left px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-zinc-400">
                    Name
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-zinc-400">
                    Cases
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-zinc-400">
                    Updated
                  </th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-zinc-400">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr
                    key={d.id}
                    onClick={() => navigate(`/datasets/${d.id}`)}
                    className="border-b border-zinc-100 last:border-b-0 cursor-pointer hover:bg-zinc-50 transition-colors"
                  >
                    <td className="px-4 py-3.5">
                      <div className="font-medium">{d.name}</div>
                      {d.description && (
                        <div className="text-sm text-zinc-500 mt-0.5">{d.description}</div>
                      )}
                    </td>
                    <td className="px-4 py-3.5 font-mono text-sm">{d.case_count}</td>
                    <td className="px-4 py-3.5 text-zinc-500 text-sm">{relativeTime(d.updated_at)}</td>
                    <td className="px-4 py-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                      <Button variant="ghost" size="sm" asChild>
                        <Link to={`/datasets/${d.id}`}>View</Link>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-500 hover:text-red-600 hover:bg-red-50"
                        onClick={() =>
                          setShowDeleteModal({ id: d.id, name: d.name, count: d.case_count })
                        }
                      >
                        Delete
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Delete confirmation modal */}
      <Dialog open={!!showDeleteModal} onOpenChange={() => setShowDeleteModal(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Dataset</DialogTitle>
          </DialogHeader>
          <p>
            Are you sure you want to delete <strong>{showDeleteModal?.name}</strong>? This will
            permanently remove the dataset and all {showDeleteModal?.count} cases. This action
            cannot be undone.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteModal(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              Delete Dataset
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* New Dataset Modal */}
      <NewDatasetModal
        open={showNewDatasetModal}
        onOpenChange={setShowNewDatasetModal}
        onCreated={(dataset) => {
          setShowNewDatasetModal(false)
          loadDatasets()
          navigate(`/datasets/${dataset.id}`)
        }}
      />
    </>
  )
}

function NewDatasetModal({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated: (dataset: Dataset) => void
}) {
  const { showToast } = useStore()
  const [mode, setMode] = useState<'generate' | 'manual'>('generate')
  const [productDescription, setProductDescription] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (open) {
      setMode('generate')
      setProductDescription('')
      setName('')
      setDescription('')
    }
  }, [open])

  const handleGenerate = async () => {
    if (!productDescription.trim()) {
      showToast('Please describe your product', true)
      return
    }

    setLoading(true)
    try {
      const dataset = await api.generateDataset(productDescription)
      showToast(`Created dataset "${dataset.name}" with ${dataset.case_count} cases`)
      onCreated(dataset)
    } catch (e) {
      showToast((e as Error).message || 'Failed to generate dataset', true)
    } finally {
      setLoading(false)
    }
  }

  const handleCreate = async () => {
    if (!name.trim()) {
      showToast('Please enter a dataset name', true)
      return
    }

    setLoading(true)
    try {
      const dataset = await api.createDataset({
        name: name.trim(),
        description: description.trim() || undefined,
      })
      showToast('Dataset created')
      onCreated(dataset)
    } catch (e) {
      showToast((e as Error).message || 'Failed to create dataset', true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>New Dataset</DialogTitle>
        </DialogHeader>

        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setMode('generate')}
            className={`flex-1 px-3 py-2 text-sm font-medium border transition-colors ${mode === 'generate'
              ? 'border-taro bg-taro/10 text-taro'
              : 'border-zinc-200 text-zinc-600 hover:border-zinc-300'
              }`}
          >
            Generate with AI
          </button>
          <button
            onClick={() => setMode('manual')}
            className={`flex-1 px-3 py-2 text-sm font-medium border transition-colors ${mode === 'manual'
              ? 'border-taro bg-taro/10 text-taro'
              : 'border-zinc-200 text-zinc-600 hover:border-zinc-300'
              }`}
          >
            Manual
          </button>
        </div>

        {mode === 'generate' ? (
          <div>
            <label className="block text-sm font-medium text-zinc-600 mb-1.5">
              Describe your product
            </label>
            <textarea
              value={productDescription}
              onChange={(e) => setProductDescription(e.target.value)}
              rows={4}
              placeholder="A customer support chatbot for an e-commerce site that handles order inquiries, returns, and shipping questions..."
              className="w-full px-3 py-2 border border-zinc-200 bg-white text-sm focus:outline-none focus:ring-1 focus:ring-taro resize-y"
            />
            <p className="text-xs text-zinc-400 mt-1.5">
              We'll generate a dataset based on your product and what it does.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-zinc-600 mb-1.5">Name</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Dataset"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-600 mb-1.5">
                Description (optional)
              </label>
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="A brief description of this dataset"
              />
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={loading}>
            Cancel
          </Button>
          <Button
            onClick={mode === 'generate' ? handleGenerate : handleCreate}
            disabled={loading}
          >
            {loading
              ? mode === 'generate'
                ? 'Generating...'
                : 'Creating...'
              : mode === 'generate'
                ? 'Generate Dataset'
                : 'Create Dataset'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function DatasetDetail({
  dataset,
  cases,
  loading,
  onSelectCase,
  selectedCaseId,
  onAddCase,
}: {
  dataset: Dataset
  cases: Case[]
  loading: boolean
  onSelectCase: (c: Case) => void
  selectedCaseId?: string
  onAddCase: () => void
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <div className="mr-2 h-4 w-4 border-2 border-zinc-200 border-t-taro rounded-full animate-spin" />
        Loading...
      </div>
    )
  }

  return (
    <div>
      <Link to="/datasets" className="text-sm text-zinc-500 hover:text-zinc-900 mb-4 inline-block">
        ← Back to Datasets
      </Link>

      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{dataset.name}</h1>
          {dataset.description && <p className="text-sm text-zinc-500 mt-1">{dataset.description}</p>}
          <p className="text-sm text-zinc-400 mt-1">{cases.length} cases</p>
        </div>
        <Button onClick={onAddCase}>+ Add Case</Button>
      </div>

      {cases.length === 0 ? (
        <Card>
          <CardContent className="py-20 text-center">
            <h3 className="text-base font-semibold mb-2">No cases yet</h3>
            <p className="text-zinc-500 mb-5">Add cases manually or generate them with AI.</p>
            <Button onClick={onAddCase}>+ Add Case</Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            {cases.map((c) => {
              const name = c.name || `Case #${c.id.substring(0, 8)}`
              const messageCount = c.inputs?.length || 0

              return (
                <div
                  key={c.id}
                  onClick={() => onSelectCase(c)}
                  className={`flex items-center px-4 py-3.5 border-b border-zinc-100 last:border-b-0 cursor-pointer transition-colors ${selectedCaseId === c.id ? 'bg-taro/10' : 'hover:bg-zinc-50'
                    }`}
                >
                  <span className="flex-1 font-medium">{name}</span>
                  <span className="text-xs text-zinc-400">
                    {messageCount} message{messageCount !== 1 ? 's' : ''}
                  </span>
                  <span className="text-zinc-300 ml-2">›</span>
                </div>
              )
            })}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function CaseSidebar({
  caseData,
  onClose,
  onEdit,
  onDelete,
}: {
  caseData: Case | null
  onClose: () => void
  onEdit: (c: Case) => void
  onDelete: (id: string) => void
}) {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && caseData) {
        onClose()
      }
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [caseData, onClose])

  if (!caseData) return null

  const name = caseData.name || `Case #${caseData.id.substring(0, 8)}`

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/30 z-40" onClick={onClose} />

      {/* Sidebar */}
      <div className="fixed top-0 right-0 w-[480px] h-full bg-white border-l border-zinc-200 z-50 flex flex-col">
        <div className="flex justify-between items-center px-5 py-4 border-b border-zinc-200 bg-zinc-50">
          <span className="font-semibold">{name}</span>
          <button onClick={onClose} className="text-2xl text-zinc-400 hover:text-zinc-900">
            ×
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <div className="mb-6">
            <div className="text-xs font-medium uppercase tracking-wide text-zinc-400 mb-3">
              Conversation
            </div>
            {caseData.inputs.map((m, i) => (
              <div key={i} className="flex gap-3 mb-4 last:mb-0">
                <span
                  className={`text-xs font-semibold uppercase px-2 py-1 min-w-[70px] text-center shrink-0 ${m.role === 'user' ? 'bg-taro/10 text-taro' : 'bg-zinc-100 text-zinc-600'
                    }`}
                >
                  {m.role}
                </span>
                <div className="text-sm leading-relaxed">{m.message}</div>
              </div>
            ))}
          </div>

          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-zinc-400 mb-3">
              Expected Outcome
            </div>
            <div className="p-4 bg-zinc-50 border border-zinc-200 text-sm leading-relaxed">
              {caseData.expected_outcome}
            </div>
            {caseData.expected_source && (
              <>
                <div className="mt-3 inline-flex items-center gap-2 bg-taro/10 px-2.5 py-1 text-xs">
                  <span className="font-mono">{caseData.expected_source.file}</span>
                  <span className="bg-taro text-white px-1.5 py-0.5 text-xs font-medium">
                    p. {caseData.expected_source.page}
                  </span>
                </div>
                {caseData.expected_source.excerpt && (
                  <div className="mt-2 text-sm italic text-zinc-500">
                    "{caseData.expected_source.excerpt}"
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="px-5 py-4 border-t border-zinc-200 flex gap-2 justify-end">
          <Button
            variant="ghost"
            size="sm"
            className="text-red-500 hover:text-red-600 hover:bg-red-50"
            onClick={() => onDelete(caseData.id)}
          >
            Delete
          </Button>
          <Button variant="outline" size="sm" onClick={() => onEdit(caseData)}>
            Edit
          </Button>
        </div>
      </div>
    </>
  )
}

function CaseModal({
  open,
  onOpenChange,
  editingCase,
  datasetId,
  onSaved,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  editingCase: Case | null
  datasetId: string
  onSaved: () => void
}) {
  const { showToast } = useStore()
  const [name, setName] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; message: string }[]>([
    { role: 'user', message: '' },
  ])
  const [outcome, setOutcome] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (editingCase) {
      setName(editingCase.name || '')
      setMessages(
        editingCase.inputs
          .filter((i) => i.role === 'user' || i.role === 'assistant')
          .map((i) => ({ role: i.role as 'user' | 'assistant', message: i.message }))
      )
      setOutcome(editingCase.expected_outcome)
    } else {
      setName('')
      setMessages([{ role: 'user', message: '' }])
      setOutcome('')
    }
  }, [editingCase, open])

  const handleSave = async () => {
    const validMessages = messages.filter((m) => m.message.trim())
    if (validMessages.length === 0) {
      showToast('Please add at least one message', true)
      return
    }
    if (!outcome.trim()) {
      showToast('Please enter an expected outcome', true)
      return
    }

    setSaving(true)
    try {
      const inputs = validMessages.map((m) => ({
        role: m.role,
        message: m.message,
        attachments: [],
      }))

      if (editingCase) {
        await api.updateCase(datasetId, editingCase.id, {
          name: name || undefined,
          inputs,
          expected_outcome: outcome,
        })
        showToast('Case updated')
      } else {
        await api.createCase({
          dataset_name: datasetId,
          name: name || undefined,
          inputs,
          expected_outcome: outcome,
        })
        showToast('Case added')
      }
      onSaved()
    } catch {
      showToast('Failed to save case', true)
    } finally {
      setSaving(false)
    }
  }

  const addMessage = () => {
    setMessages([...messages, { role: 'user', message: '' }])
  }

  const removeMessage = (index: number) => {
    setMessages(messages.filter((_, i) => i !== index))
  }

  const updateMessage = (index: number, field: 'role' | 'message', value: string) => {
    const updated = [...messages]
    if (field === 'role') {
      updated[index].role = value as 'user' | 'assistant'
    } else {
      updated[index].message = value
    }
    setMessages(updated)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>{editingCase ? 'Edit Case' : 'Add Case'}</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-zinc-600 mb-1.5">
              Name (optional)
            </label>
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-600 mb-1.5">Conversation</label>
            <div className="space-y-2">
              {messages.map((m, i) => (
                <div key={i} className="flex gap-2">
                  <select
                    value={m.role}
                    onChange={(e) => updateMessage(i, 'role', e.target.value)}
                    className="w-24 px-2 py-2 border border-zinc-200 bg-white text-sm focus:outline-none focus:ring-1 focus:ring-taro"
                  >
                    <option value="user">User</option>
                    <option value="assistant">Assistant</option>
                  </select>
                  <Input
                    value={m.message}
                    onChange={(e) => updateMessage(i, 'message', e.target.value)}
                    className="flex-1"
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-red-500 hover:text-red-600 hover:bg-red-50"
                    onClick={() => removeMessage(i)}
                  >
                    ×
                  </Button>
                </div>
              ))}
            </div>
            <Button variant="outline" size="sm" className="mt-2" onClick={addMessage}>
              + Add Message
            </Button>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-600 mb-1.5">
              Expected Outcome
            </label>
            <textarea
              value={outcome}
              onChange={(e) => setOutcome(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-zinc-200 bg-white text-sm focus:outline-none focus:ring-1 focus:ring-taro resize-y"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : 'Save Case'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
