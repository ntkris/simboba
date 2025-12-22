import React, { createContext, useContext, useReducer, useCallback } from 'react'
import type { ReactNode } from 'react'
import type { Dataset, RunSummary, Settings } from '@/types'
import * as api from '@/lib/api'

interface State {
  datasets: Dataset[]
  runs: RunSummary[]
  settings: Settings | null
  loading: {
    datasets: boolean
    runs: boolean
    settings: boolean
  }
  error: string | null
}

type Action =
  | { type: 'SET_DATASETS'; payload: Dataset[] }
  | { type: 'SET_RUNS'; payload: RunSummary[] }
  | { type: 'SET_SETTINGS'; payload: Settings }
  | { type: 'SET_LOADING'; payload: { key: keyof State['loading']; value: boolean } }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'ADD_DATASET'; payload: Dataset }
  | { type: 'UPDATE_DATASET'; payload: Dataset }
  | { type: 'REMOVE_DATASET'; payload: string }

const initialState: State = {
  datasets: [],
  runs: [],
  settings: null,
  loading: {
    datasets: false,
    runs: false,
    settings: false,
  },
  error: null,
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'SET_DATASETS':
      return { ...state, datasets: action.payload }
    case 'SET_RUNS':
      return { ...state, runs: action.payload }
    case 'SET_SETTINGS':
      return { ...state, settings: action.payload }
    case 'SET_LOADING':
      return { ...state, loading: { ...state.loading, [action.payload.key]: action.payload.value } }
    case 'SET_ERROR':
      return { ...state, error: action.payload }
    case 'ADD_DATASET':
      return { ...state, datasets: [action.payload, ...state.datasets] }
    case 'UPDATE_DATASET':
      return {
        ...state,
        datasets: state.datasets.map(d => d.id === action.payload.id ? action.payload : d),
      }
    case 'REMOVE_DATASET':
      return { ...state, datasets: state.datasets.filter(d => d.id !== action.payload) }
    default:
      return state
  }
}

interface StoreContextType {
  state: State
  loadDatasets: () => Promise<void>
  loadRuns: () => Promise<void>
  loadSettings: () => Promise<void>
  refreshAll: () => Promise<void>
  createDataset: (name: string, description?: string) => Promise<Dataset>
  deleteDataset: (id: string) => Promise<void>
  deleteRun: (datasetId: string, filename: string) => Promise<void>
  updateSettings: (settings: Partial<Settings>) => Promise<void>
  showToast: (message: string, isError?: boolean) => void
}

const StoreContext = createContext<StoreContextType | null>(null)

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [toast, setToast] = React.useState<{ message: string; isError: boolean } | null>(null)

  const showToast = useCallback((message: string, isError = false) => {
    setToast({ message, isError })
    setTimeout(() => setToast(null), 3000)
  }, [])

  const loadDatasets = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'datasets', value: true } })
    try {
      const datasets = await api.listDatasets()
      dispatch({ type: 'SET_DATASETS', payload: datasets })
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message })
      showToast('Failed to load datasets', true)
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'datasets', value: false } })
    }
  }, [showToast])

  const loadRuns = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'runs', value: true } })
    try {
      const runs = await api.listRuns()
      dispatch({ type: 'SET_RUNS', payload: runs })
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message })
      showToast('Failed to load runs', true)
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'runs', value: false } })
    }
  }, [showToast])

  const loadSettings = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'settings', value: true } })
    try {
      const settings = await api.getSettings()
      dispatch({ type: 'SET_SETTINGS', payload: settings })
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message })
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'settings', value: false } })
    }
  }, [])

  const refreshAll = useCallback(async () => {
    await Promise.all([loadDatasets(), loadRuns(), loadSettings()])
  }, [loadDatasets, loadRuns, loadSettings])

  const createDataset = useCallback(async (name: string, description?: string) => {
    const dataset = await api.createDataset({ name, description })
    dispatch({ type: 'ADD_DATASET', payload: dataset })
    showToast('Dataset created')
    return dataset
  }, [showToast])

  const deleteDatasetAction = useCallback(async (id: string) => {
    await api.deleteDataset(id)
    dispatch({ type: 'REMOVE_DATASET', payload: id })
    showToast('Dataset deleted')
  }, [showToast])

  const deleteRunAction = useCallback(async (datasetId: string, filename: string) => {
    await api.deleteRun(datasetId, filename)
    await loadRuns()
    showToast('Run deleted')
  }, [loadRuns, showToast])

  const updateSettingsAction = useCallback(async (settings: Partial<Settings>) => {
    const updated = await api.updateSettings(settings)
    dispatch({ type: 'SET_SETTINGS', payload: updated })
    showToast('Settings saved')
  }, [showToast])

  return (
    <StoreContext.Provider
      value={{
        state,
        loadDatasets,
        loadRuns,
        loadSettings,
        refreshAll,
        createDataset,
        deleteDataset: deleteDatasetAction,
        deleteRun: deleteRunAction,
        updateSettings: updateSettingsAction,
        showToast,
      }}
    >
      {children}
      {toast && (
        <div
          className={`fixed bottom-4 right-4 px-4 py-2 text-sm text-white z-50 ${
            toast.isError ? 'bg-error' : 'bg-zinc-800'
          }`}
        >
          {toast.message}
        </div>
      )}
    </StoreContext.Provider>
  )
}

export function useStore() {
  const context = useContext(StoreContext)
  if (!context) {
    throw new Error('useStore must be used within StoreProvider')
  }
  return context
}
