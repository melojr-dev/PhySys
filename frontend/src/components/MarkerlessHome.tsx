import React from "react";
import type { AnalysisResult } from "../types";

function fmt(value: number | null | undefined, digits = 1) {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

export function MarkerlessHome({
  onStart,
  onOpenResearch,
  result,
}: {
  onStart: () => void;
  onOpenResearch: () => void;
  result: AnalysisResult | null;
}) {
  return (
    <main className="markerless-home">
      <section className="markerless-hero panel">
        <div className="markerless-hero-copy">
          <div className="markerless-kicker"><span className="markerless-pulse" /> MARKERLESS CORE</div>
          <h1>Análise de movimento humano a partir de <em>vídeo comum</em>.</h1>
          <p>
            O PhysioAI identifica pontos corporais sem marcadores físicos, acompanha o movimento frame a frame
            e transforma vídeo em métricas biomecânicas auditáveis.
          </p>
          <div className="markerless-hero-actions">
            <button className="primary-button hero-button" onClick={onStart}>Iniciar análise markerless</button>
            <button className="secondary-button hero-button" onClick={onOpenResearch}>Abrir área de pesquisa</button>
          </div>
          <div className="markerless-promises" aria-label="Características principais">
            <span><strong>01</strong> Sem marcadores corporais</span>
            <span><strong>02</strong> Sem sensor vestível obrigatório</span>
            <span><strong>03</strong> Medição por vídeo</span>
          </div>
        </div>

        <div className="markerless-hero-visual" aria-hidden="true">
          <div className="vision-frame">
            <div className="vision-grid" />
            <svg viewBox="0 0 420 520" className="hero-skeleton">
              <path d="M206 94 L203 155 L164 216 L132 282" />
              <path d="M203 155 L251 211 L294 270" />
              <path d="M203 155 L202 290 L165 392 L150 480" />
              <path d="M202 290 L246 390 L268 480" />
              <path d="M164 216 L132 282 L118 330" className="hero-highlight" />
              {[ [206,94],[203,155],[164,216],[132,282],[118,330],[251,211],[294,270],[202,290],[165,392],[150,480],[246,390],[268,480] ].map(([x,y], i) => <circle key={i} cx={x} cy={y} r={i >= 2 && i <= 4 ? 8 : 6} className={i >= 2 && i <= 4 ? "hero-node active" : "hero-node"} />)}
              <circle cx="203" cy="155" r="42" className="hero-angle-ring" />
              <text x="226" y="144" className="hero-angle-text">82.6°</text>
            </svg>
            <div className="vision-label top">POSE TRACKING · FRAME 0084</div>
            <div className="vision-label bottom">MEMBRO DIREITO · CONFIANÇA 96%</div>
          </div>
        </div>
      </section>

      <section className="markerless-flow" aria-label="Como o PhysioAI funciona">
        {[
          ["01", "Grave", "Use uma câmera convencional com o membro totalmente visível."],
          ["02", "Detecte", "O motor markerless localiza landmarks corporais em cada frame."],
          ["03", "Meça", "Referenciais anatômicos transformam os landmarks em ângulos e séries temporais."],
          ["04", "Compare", "Resultados podem ser revisados, exportados e comparados entre sessões."],
        ].map(([n, title, text]) => (
          <article key={n} className="flow-card panel">
            <span>{n}</span><strong>{title}</strong><p>{text}</p>
          </article>
        ))}
      </section>

      <section className="home-grid">
        <article className="home-capability panel">
          <span className="eyebrow">O que o usuário vê</span>
          <h2>Do vídeo à compreensão do movimento</h2>
          <div className="capability-stack">
            <div><span className="capability-dot" /><strong>Overlay anatômico</strong><p>O membro selecionado é acompanhado no próprio vídeo.</p></div>
            <div><span className="capability-dot" /><strong>Goniometria digital</strong><p>Ângulo clínico, amplitude e velocidade sincronizados ao frame.</p></div>
            <div><span className="capability-dot" /><strong>Modelo corporal 3D</strong><p>Representação virtual estimada para explorar postura e movimento.</p></div>
            <div><span className="capability-dot" /><strong>Rastreabilidade científica</strong><p>Modo avançado preserva protocolo, confiança e proveniência.</p></div>
          </div>
        </article>

        <article className="home-last-result panel">
          <span className="eyebrow">Sessão atual</span>
          <h2>{result ? "Resultado disponível" : "Nenhuma análise nesta sessão"}</h2>
          {result ? (
            <>
              <div className="home-result-number"><strong>{fmt(result.adm_observada ?? result.adm_max)}</strong><span>° ADM observada</span></div>
              <div className="home-result-metrics">
                <div><span>Ângulo médio</span><strong>{fmt(result.angulo_clinico_medio)}°</strong></div>
                <div><span>Vel. média</span><strong>{fmt(result.vel_media)}°/s</strong></div>
                <div><span>Repetições</span><strong>{result.total_reps ?? "—"}</strong></div>
              </div>
              <button className="secondary-button" onClick={onStart}>Reabrir avaliação</button>
            </>
          ) : (
            <>
              <div className="empty-session-orbit"><span /><span /><span /></div>
              <p>Comece por uma captura markerless. O vídeo permanece como referência visual durante toda a análise.</p>
              <button className="primary-button" onClick={onStart}>Nova análise</button>
            </>
          )}
        </article>
      </section>

      <section className="markerless-disclaimer panel">
        <div><span className="eyebrow">Escopo científico</span><strong>Markerless é o núcleo; sensores são extensões opcionais.</strong></div>
        <p>IMU, profundidade/LiDAR e módulos experimentais complementam a plataforma, mas não são necessários para a análise markerless principal. O PhysioAI permanece uma ferramenta de apoio à avaliação e pesquisa, não um sistema diagnóstico autônomo.</p>
      </section>
    </main>
  );
}
