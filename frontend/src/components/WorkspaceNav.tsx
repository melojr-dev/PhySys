import React from "react";

export type WorkspaceKey = "home" | "assessment" | "studies" | "experimental" | "research" | "admin";

const items: Array<{ key: WorkspaceKey; label: string; short: string; description: string; admin?: boolean }> = [
  { key: "home", label: "Início", short: "IN", description: "Visão geral markerless" },
  { key: "assessment", label: "Avaliação", short: "AV", description: "Vídeo, movimento e resultado" },
  { key: "studies", label: "Estudos", short: "ST", description: "Validação e execução formal" },
  { key: "experimental", label: "Laboratório", short: "LB", description: "Marcha, IMU e módulos 3D" },
  { key: "research", label: "Pesquisa", short: "RS", description: "Sessões, dataset e exportações" },
  { key: "admin", label: "Administração", short: "AD", description: "Sistema, auditoria e LGPD", admin: true },
];

export function WorkspaceNav({ active, onChange, isAdmin }: { active: WorkspaceKey; onChange: (key: WorkspaceKey) => void; isAdmin: boolean }) {
  return (
    <nav className="workspace-nav" aria-label="Módulos PhysioAI">
      {items.filter((item) => !item.admin || isAdmin).map((item) => (
        <button
          key={item.key}
          className={`workspace-nav-item ${active === item.key ? "active" : ""}`}
          onClick={() => onChange(item.key)}
          type="button"
        >
          <span className="workspace-nav-icon" aria-hidden="true">{item.short}</span>
          <span className="workspace-nav-copy"><strong>{item.label}</strong><small>{item.description}</small></span>
        </button>
      ))}
    </nav>
  );
}
