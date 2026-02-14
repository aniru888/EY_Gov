"use client";

import { useState } from "react";
import Link from "next/link";
import NavLink from "./NavLink";

export default function Header() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div>
            <Link href="/" className="flex flex-col">
              <span className="text-lg font-bold text-gray-900 leading-tight">
                State Economic Activity Index
              </span>
              <span className="text-xs text-gray-500">
                Li Keqiang Index for Indian States
              </span>
            </Link>
          </div>

          {/* Desktop nav */}
          <nav className="hidden sm:flex gap-6 text-sm font-medium">
            <NavLink href="/" exact>Home</NavLink>
            <NavLink href="/rankings">Rankings</NavLink>
            <NavLink href="/electricity">Electricity</NavLink>
            <NavLink href="/insights">Insights</NavLink>
            <NavLink href="/methodology">Methodology</NavLink>
            <NavLink href="/compare">Compare</NavLink>
          </nav>

          {/* Mobile hamburger */}
          <button
            className="sm:hidden p-2 text-gray-600 hover:text-gray-900"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              {menuOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="sm:hidden border-t border-gray-200 bg-white px-4 pb-3 pt-2">
          <div className="flex flex-col gap-2 text-sm font-medium">
            <NavLink href="/" exact onClick={() => setMenuOpen(false)}>
              Home
            </NavLink>
            <NavLink href="/rankings" onClick={() => setMenuOpen(false)}>
              Rankings
            </NavLink>
            <NavLink href="/electricity" onClick={() => setMenuOpen(false)}>
              Electricity
            </NavLink>
            <NavLink href="/insights" onClick={() => setMenuOpen(false)}>
              Insights
            </NavLink>
            <NavLink href="/methodology" onClick={() => setMenuOpen(false)}>
              Methodology
            </NavLink>
            <NavLink href="/compare" onClick={() => setMenuOpen(false)}>
              Compare
            </NavLink>
          </div>
        </div>
      )}
    </header>
  );
}
