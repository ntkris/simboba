import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { Datasets } from '@/pages/Datasets'
import { Runs } from '@/pages/Runs'
import { Settings } from '@/pages/Settings'
import { useStore } from '@/hooks/useStore'
import { useEffect } from 'react'

function App() {
  const { refreshAll } = useStore()

  useEffect(() => {
    refreshAll()
  }, [refreshAll])

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/datasets/:id" element={<Datasets />} />
        <Route path="/runs" element={<Runs />} />
        <Route path="/runs/:datasetId/:filename" element={<Runs />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App
