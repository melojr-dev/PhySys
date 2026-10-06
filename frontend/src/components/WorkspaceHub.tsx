import React, { useState } from "react";
import type { AnalysisResult, AuthUser } from "../types";
import type { WorkspaceKey } from "./WorkspaceNav";
import { FormalValidationPanel } from "./FormalValidationPanel";
import { StudyManagementPanel } from "./StudyManagementPanel";
import { GaitPanel } from "./GaitPanel";
import { SensorFusionPanel } from "./SensorFusionPanel";
import { MotorFeaturesPanel } from "./MotorFeaturesPanel";
import { LongitudinalPanel } from "./LongitudinalPanel";
import { ResearchWorkspace } from "./ResearchWorkspace";
import { AdminWorkspace } from "./AdminWorkspace";

export function WorkspaceHub({
  active,
  user,
  jobId,
  result,
  imuAnalysisId,
  onImuAnalysis,
}: {
  active: Exclude<WorkspaceKey, "home" | "assessment">;
  user: AuthUser;
  jobId: string | null;
  result: AnalysisResult | null;
  imuAnalysisId: string | null;
  onImuAnalysis: (id: string) => void;
}) {
  if (active === "studies") {
    return <div className="portal-page"><div className="portal-page-header"><div><span className="eyebrow">Estudos</span><h1>Validação e coleta formal</h1><p>Governança de protocolo, execução formal e preparação do pacote de revisão externa.</p></div></div><StudyManagementPanel isAdmin={user.role === "admin"} /><FormalValidationPanel isAdmin={user.role === "admin"} /></div>;
  }

  if (active === "experimental") return <ExperimentalWorkspace jobId={jobId} result={result} imuAnalysisId={imuAnalysisId} onImuAnalysis={onImuAnalysis} />;
  if (active === "research") return <ResearchWorkspace />;
  if (active === "admin") return <AdminWorkspace />;
  return null;
}

function ExperimentalWorkspace({ jobId, result, imuAnalysisId, onImuAnalysis }: { jobId: string | null; result: AnalysisResult | null; imuAnalysisId: string | null; onImuAnalysis: (id: string) => void }) {
  const [tab, setTab] = useState<"gait" | "imu" | "motor" | "longitudinal">("imu");
  return <div className="portal-page">
    <div className="portal-page-header"><div><span className="eyebrow">Experimental</span><h1>Análises de pesquisa</h1><p>Módulos descritivos e não diagnósticos, separados da via clínica principal.</p></div><div className="segmented-control wrap"><button className={tab === "gait" ? "active" : ""} onClick={() => setTab("gait")}>Marcha</button><button className={tab === "imu" ? "active" : ""} onClick={() => setTab("imu")}>IMU</button><button className={tab === "motor" ? "active" : ""} onClick={() => setTab("motor")}>Perfil motor</button><button className={tab === "longitudinal" ? "active" : ""} onClick={() => setTab("longitudinal")}>Longitudinal</button></div></div>
    <div className="context-ribbon"><span>Sessão ativa <strong>{jobId ? jobId.slice(0, 12) : "nenhuma"}</strong></span><span>IMU vinculada <strong>{imuAnalysisId ? imuAnalysisId.slice(0, 12) : "nenhuma"}</strong></span><span>Resultado <strong>{result ? "disponível" : "não carregado"}</strong></span></div>
    {tab === "gait" && <GaitPanel result={result} />}
    {tab === "imu" && <SensorFusionPanel jobId={jobId} onAnalysis={(record) => onImuAnalysis(record.id)} />}
    {tab === "motor" && <MotorFeaturesPanel jobId={jobId} imuAnalysisId={imuAnalysisId} />}
    {tab === "longitudinal" && <LongitudinalPanel jobId={jobId} />}
  </div>;
}
