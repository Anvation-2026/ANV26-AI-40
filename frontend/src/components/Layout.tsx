import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { SidebarNav } from './SidebarNav';
import { TopBar } from './TopBar';
import { getHealth, getModelStatus } from '../services/api';
import { ModelStatusResponse } from '../types/analysis';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState<boolean>(false);
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);
  const [backendAlive, setBackendAlive] = useState<boolean | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then(() => {
        if (mounted) setBackendAlive(true);
      })
      .catch(() => {
        if (mounted) setBackendAlive(false);
      });

    getModelStatus()
      .then((data) => {
        if (mounted) setModelStatus(data);
      })
      .catch(() => {});

    return () => {
      mounted = false;
    };
  }, [location.pathname]);

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
        return 'Clinical Overview & Architecture';
      case '/analyze':
        return 'Decision Support & Triage Workspace';
      case '/validation':
        return 'Model Validation Benchmark';
      case '/about':
        return 'Methodology & Trust Boundaries';
      default:
        return 'MedGuard AI Decision Support';
    }
  };

  return (
    <div className="flex h-screen bg-canvas overflow-hidden text-navy-foreground font-sans">
      {/* Desktop Collapsible Sidebar */}
      <div className="hidden md:flex flex-shrink-0">
        <SidebarNav
          collapsed={collapsed}
          onToggleCollapse={() => setCollapsed(!collapsed)}
          modelStatus={modelStatus}
        />
      </div>

      {/* Mobile Sidebar Overlay Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden">
          <div
            className="fixed inset-0 bg-navy/60 backdrop-blur-xs transition-opacity"
            onClick={() => setMobileOpen(false)}
          />
          <div className="relative flex-1 flex flex-col max-w-xs w-full bg-navy z-10">
            <SidebarNav
              collapsed={false}
              onToggleCollapse={() => setMobileOpen(false)}
              modelStatus={modelStatus}
            />
          </div>
        </div>
      )}

      {/* Main Content Region */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar
          pageTitle={getPageTitle()}
          backendAlive={backendAlive}
          onToggleMobileNav={() => setMobileOpen(true)}
        />

        <main className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="max-w-7xl mx-auto w-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};
