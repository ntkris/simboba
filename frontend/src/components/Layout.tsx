import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/datasets', label: 'Datasets' },
  { to: '/runs', label: 'Runs' },
  { to: '/settings', label: 'Settings' },
]

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <rect x="4" y="4" width="20" height="20" rx="4" fill="#8B7BA5" />
        <circle cx="10" cy="14" r="2" fill="#18181b" />
        <circle cx="14" cy="18" r="2" fill="#18181b" />
        <circle cx="18" cy="15" r="2" fill="#18181b" />
      </svg>
      <span className="font-semibold text-zinc-900">simboba</span>
    </div>
  )
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-zinc-50">
      <header className="h-14 border-b border-zinc-200 bg-white px-4 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Logo />
          <nav className="flex items-center gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  cn(
                    'px-3 py-1.5 text-sm transition-colors',
                    isActive
                      ? 'bg-zinc-100 text-zinc-900 font-medium'
                      : 'text-zinc-600 hover:text-zinc-900 hover:bg-zinc-50'
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="p-6">{children}</main>
    </div>
  )
}
