import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Clipboard,
  Cpu,
  Database,
  Download,
  FolderOpen,
  Gauge,
  HardDrive,
  Home,
  Layers,
  Loader2,
  Play,
  Settings,
  SlidersHorizontal,
  Terminal,
  Upload,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";

type PageId = "dashboard" | "inference" | "train" | "benchmark" | "models" | "settings";

type ModelInfo = {
  name: string;
  path: string;
  sizeMb: number;
  quantization: string;
  parameters: string;
  modifiedMs: number;
};

type SystemStats = {
  memoryMb: number;
  coldStartMs: number;
  binaryMb: number;
  modelsDir: string;
  binaryPath: string;
};

type InferResult = {
  text: string;
  tokensGenerated: number;
  tokensPerSecond: number;
  timeMs: number;
};

type TrainConfig = {
  baseModel: string;
  dataset: string;
  outputDir: string;
  loraRank: number;
  learningRate: number;
  maxSteps: number;
  batchSize: number;
};

type TrainJob = {
  jobId: string;
  status: string;
  pid?: number;
};

type TrainStatus = {
  jobId: string;
  status: string;
  step: number;
  maxSteps: number;
  loss?: number;
  progress: number;
  elapsedSeconds: number;
  logTail: string;
};

type BenchmarkResult = {
  model: string;
  tokensPerSecond: number;
  coldStartMs: number;
  memoryMb: number;
  runs: number;
};

type ActivityItem = {
  id: string;
  title: string;
  detail: string;
  tone: "ok" | "warn" | "info";
};

type IconType = LucideIcon;

const navItems: Array<{ id: PageId; label: string; icon: IconType }> = [
  { id: "dashboard", label: "Dashboard", icon: Home },
  { id: "inference", label: "Inference", icon: Brain },
  { id: "train", label: "Train", icon: Layers },
  { id: "benchmark", label: "Benchmark", icon: BarChart3 },
  { id: "models", label: "Models", icon: Database },
  { id: "settings", label: "Settings", icon: Settings },
];

const defaultStats: SystemStats = {
  memoryMb: 81,
  coldStartMs: 10.1,
  binaryMb: 11.44,
  modelsDir: "~/.gwenland/models",
  binaryPath: "Bundled GwenLand sidecar",
};

const defaultTrainConfig: TrainConfig = {
  baseModel: "",
  dataset: "",
  outputDir: "",
  loraRank: 8,
  learningRate: 0.0002,
  maxSteps: 800,
  batchSize: 1,
};

const quickPrompts = [
  "Summarize the model card in three deployment notes.",
  "Write a compact Rust function that parses a CLI flag.",
  "Explain the tradeoff between Q4_K_M and Q8_0 quantization.",
];

async function safeInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return invoke<T>(command, args);
}

function App() {
  const [page, setPage] = useState<PageId>("dashboard");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [stats, setStats] = useState<SystemStats>(defaultStats);
  const [selectedModel, setSelectedModel] = useState("");
  const [activity, setActivity] = useState<ActivityItem[]>([
    {
      id: "boot",
      title: "Core GUI initialized",
      detail: "Waiting for local GwenLand runtime activity.",
      tone: "info",
    },
  ]);
  const [modelError, setModelError] = useState("");

  const [prompt, setPrompt] = useState("Give me a deployment checklist for a local GGUF model.");
  const [systemPrompt, setSystemPrompt] = useState("You are GwenLand Core, a local AI toolkit.");
  const [maxTokens, setMaxTokens] = useState(160);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.95);
  const [inferenceResult, setInferenceResult] = useState<InferResult | null>(null);
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [inferenceError, setInferenceError] = useState("");

  const [trainConfig, setTrainConfig] = useState<TrainConfig>(defaultTrainConfig);
  const [trainJobs, setTrainJobs] = useState<TrainStatus[]>([]);
  const [trainBusy, setTrainBusy] = useState(false);
  const [trainError, setTrainError] = useState("");

  const [benchmarkModels, setBenchmarkModels] = useState<string[]>([]);
  const [benchmarkRuns, setBenchmarkRuns] = useState(5);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState("");

  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadBusy, setDownloadBusy] = useState(false);

  const selectedModelPath = selectedModel || models[0]?.path || "";
  const activeJob = trainJobs.find((job) => job.status === "running") ?? trainJobs[0];

  const modelOptions = useMemo(
    () =>
      models.map((model) => ({
        value: model.path,
        label: `${model.name} (${model.quantization})`,
      })),
    [models],
  );

  async function refreshRuntime() {
    try {
      const [nextStats, nextModels] = await Promise.all([
        safeInvoke<SystemStats>("get_system_stats"),
        safeInvoke<ModelInfo[]>("list_models"),
      ]);
      setStats(nextStats);
      setModels(nextModels);
      setModelError("");
      if (!selectedModel && nextModels[0]) {
        setSelectedModel(nextModels[0].path);
      }
    } catch (error) {
      setModelError(readError(error));
    }
  }

  function pushActivity(item: Omit<ActivityItem, "id">) {
    setActivity((current) => [
      { ...item, id: `${Date.now()}-${item.title}` },
      ...current,
    ].slice(0, 8));
  }

  useEffect(() => {
    refreshRuntime();
  }, []);

  useEffect(() => {
    const runningJobs = trainJobs.filter((job) => job.status === "running");
    if (runningJobs.length === 0) return;

    const timer = window.setInterval(async () => {
      const updates = await Promise.all(
        runningJobs.map((job) =>
          safeInvoke<TrainStatus>("get_train_status", { jobId: job.jobId }).catch(() => job),
        ),
      );
      setTrainJobs((current) =>
        current.map((job) => updates.find((update) => update.jobId === job.jobId) ?? job),
      );
    }, 1500);

    return () => window.clearInterval(timer);
  }, [trainJobs]);

  async function runInference() {
    if (!selectedModelPath) {
      setInferenceError("Import or select a model before running inference.");
      return;
    }
    setInferenceBusy(true);
    setInferenceError("");
    try {
      const result = await safeInvoke<InferResult>("run_inference", {
        modelPath: selectedModelPath,
        prompt,
        params: { maxTokens, temperature, topP, systemPrompt },
      });
      setInferenceResult(result);
      pushActivity({
        title: "Inference completed",
        detail: `${formatNumber(result.tokensPerSecond)} tok/s on ${shortName(selectedModelPath)}`,
        tone: "ok",
      });
    } catch (error) {
      setInferenceError(readError(error));
    } finally {
      setInferenceBusy(false);
    }
  }

  async function pickDataset() {
    const selected = await openDialog({
      multiple: false,
      directory: false,
      filters: [{ name: "Datasets", extensions: ["jsonl", "json", "txt", "csv", "parquet"] }],
    });
    if (typeof selected === "string") {
      setTrainConfig((config) => ({ ...config, dataset: selected }));
    }
  }

  async function pickOutputDir() {
    const selected = await openDialog({ multiple: false, directory: true });
    if (typeof selected === "string") {
      setTrainConfig((config) => ({ ...config, outputDir: selected }));
    }
  }

  async function startTraining() {
    const baseModel = trainConfig.baseModel || selectedModelPath;
    setTrainBusy(true);
    setTrainError("");
    try {
      const job = await safeInvoke<TrainJob>("start_training", {
        config: { ...trainConfig, baseModel },
      });
      const status: TrainStatus = {
        jobId: job.jobId,
        status: job.status,
        step: 0,
        maxSteps: trainConfig.maxSteps,
        progress: 0,
        elapsedSeconds: 0,
        logTail: job.pid ? `Started GwenLand training process ${job.pid}.` : "Training started.",
      };
      setTrainJobs((current) => [status, ...current]);
      pushActivity({
        title: "Training started",
        detail: `${shortName(baseModel)} -> ${shortPath(trainConfig.outputDir)}`,
        tone: "info",
      });
    } catch (error) {
      setTrainError(readError(error));
    } finally {
      setTrainBusy(false);
    }
  }

  function toggleBenchmarkModel(path: string) {
    setBenchmarkModels((current) =>
      current.includes(path) ? current.filter((item) => item !== path) : [...current, path],
    );
  }

  async function runBenchmark() {
    const targets = benchmarkModels.length > 0 ? benchmarkModels : selectedModelPath ? [selectedModelPath] : [];
    if (targets.length === 0) {
      setBenchmarkError("Select at least one model before running benchmarks.");
      return;
    }

    setBenchmarkBusy(true);
    setBenchmarkError("");
    setBenchmarkResults([]);
    const results: BenchmarkResult[] = [];
    try {
      for (const modelPath of targets) {
        const result = await safeInvoke<BenchmarkResult>("run_benchmark", {
          modelPath,
          runs: benchmarkRuns,
        });
        results.push(result);
        setBenchmarkResults([...results]);
      }
      pushActivity({
        title: "Benchmark completed",
        detail: `${results.length} model${results.length === 1 ? "" : "s"} measured locally.`,
        tone: "ok",
      });
    } catch (error) {
      setBenchmarkError(readError(error));
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function importModel() {
    const selected = await openDialog({
      multiple: false,
      directory: false,
      filters: [{ name: "GwenLand Models", extensions: ["gguf", "ggqr"] }],
    });
    if (typeof selected !== "string") return;

    try {
      const imported = await safeInvoke<ModelInfo>("import_model", { sourcePath: selected });
      await refreshRuntime();
      setSelectedModel(imported.path);
      pushActivity({
        title: "Model imported",
        detail: imported.name,
        tone: "ok",
      });
    } catch (error) {
      setModelError(readError(error));
    }
  }

  async function downloadModel() {
    if (!downloadUrl.trim()) return;
    setDownloadBusy(true);
    setModelError("");
    try {
      const model = await safeInvoke<ModelInfo>("download_model", { url: downloadUrl.trim() });
      await refreshRuntime();
      setSelectedModel(model.path);
      setDownloadUrl("");
      pushActivity({
        title: "Model downloaded",
        detail: model.name,
        tone: "ok",
      });
    } catch (error) {
      setModelError(readError(error));
    } finally {
      setDownloadBusy(false);
    }
  }

  function updateTrainConfig<K extends keyof TrainConfig>(key: K, value: TrainConfig[K]) {
    setTrainConfig((config) => ({ ...config, [key]: value }));
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">GL</div>
          <div>
            <strong>GwenLand</strong>
            <span>Core GUI</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="GwenLand Core navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-item ${page === item.id ? "active" : ""}`}
                type="button"
                onClick={() => setPage(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="runtime-dot" />
          <div>
            <span>Local runtime</span>
            <strong>{models.length} model{models.length === 1 ? "" : "s"}</strong>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <Topbar
          page={page}
          selectedModel={selectedModelPath}
          modelOptions={modelOptions}
          onModelChange={setSelectedModel}
          onRefresh={refreshRuntime}
        />

        {page === "dashboard" && (
          <DashboardPage
            stats={stats}
            models={models}
            activity={activity}
            activeJob={activeJob}
            onNavigate={setPage}
          />
        )}
        {page === "inference" && (
          <InferencePage
            modelOptions={modelOptions}
            selectedModel={selectedModelPath}
            setSelectedModel={setSelectedModel}
            prompt={prompt}
            setPrompt={setPrompt}
            systemPrompt={systemPrompt}
            setSystemPrompt={setSystemPrompt}
            maxTokens={maxTokens}
            setMaxTokens={setMaxTokens}
            temperature={temperature}
            setTemperature={setTemperature}
            topP={topP}
            setTopP={setTopP}
            inferenceBusy={inferenceBusy}
            inferenceError={inferenceError}
            inferenceResult={inferenceResult}
            onRun={runInference}
          />
        )}
        {page === "train" && (
          <TrainPage
            config={{ ...trainConfig, baseModel: trainConfig.baseModel || selectedModelPath }}
            modelOptions={modelOptions}
            jobs={trainJobs}
            busy={trainBusy}
            error={trainError}
            onConfigChange={updateTrainConfig}
            onPickDataset={pickDataset}
            onPickOutputDir={pickOutputDir}
            onStart={startTraining}
          />
        )}
        {page === "benchmark" && (
          <BenchmarkPage
            models={models}
            selectedModels={benchmarkModels}
            runs={benchmarkRuns}
            results={benchmarkResults}
            busy={benchmarkBusy}
            error={benchmarkError}
            onToggleModel={toggleBenchmarkModel}
            onRunsChange={setBenchmarkRuns}
            onRun={runBenchmark}
          />
        )}
        {page === "models" && (
          <ModelsPage
            models={models}
            selectedModel={selectedModelPath}
            modelError={modelError}
            downloadUrl={downloadUrl}
            downloadBusy={downloadBusy}
            onSelect={setSelectedModel}
            onImport={importModel}
            onDownload={downloadModel}
            onDownloadUrlChange={setDownloadUrl}
            onBenchmark={(path) => {
              setBenchmarkModels([path]);
              setPage("benchmark");
            }}
          />
        )}
        {page === "settings" && <SettingsPage stats={stats} />}
      </main>
    </div>
  );
}

function Topbar({
  page,
  selectedModel,
  modelOptions,
  onModelChange,
  onRefresh,
}: {
  page: PageId;
  selectedModel: string;
  modelOptions: Array<{ value: string; label: string }>;
  onModelChange: (value: string) => void;
  onRefresh: () => void;
}) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">GwenLand Core</p>
        <h1>{titleForPage(page)}</h1>
      </div>
      <div className="topbar-actions">
        <select
          className="select model-select"
          value={selectedModel}
          onChange={(event) => onModelChange(event.currentTarget.value)}
          aria-label="Active model"
        >
          <option value="">No model selected</option>
          {modelOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button className="button ghost" type="button" onClick={onRefresh}>
          <Activity size={16} />
          Refresh
        </button>
      </div>
    </header>
  );
}

function DashboardPage({
  stats,
  models,
  activity,
  activeJob,
  onNavigate,
}: {
  stats: SystemStats;
  models: ModelInfo[];
  activity: ActivityItem[];
  activeJob?: TrainStatus;
  onNavigate: (page: PageId) => void;
}) {
  return (
    <div className="page-grid dashboard-grid">
      <section className="stat-grid">
        <StatCard icon={Cpu} label="Runtime Memory" value={`${formatNumber(stats.memoryMb)} MB`} hint="Core footprint" />
        <StatCard icon={Zap} label="Cold Start" value={`${formatNumber(stats.coldStartMs)} ms`} hint="Last known target" />
        <StatCard icon={HardDrive} label="Binary Size" value={`${formatNumber(stats.binaryMb)} MB`} hint="Bundled sidecar" />
        <StatCard icon={Database} label="Models" value={models.length.toString()} hint="Local registry" />
      </section>

      <Card className="hero-panel">
        <div>
          <p className="eyebrow">Standalone local toolkit</p>
          <h2>Run, train, benchmark, and manage local models without leaving the desktop.</h2>
          <p className="muted">
            The GUI talks to GwenLand Core through local Tauri commands and keeps model work on your machine.
          </p>
        </div>
        <div className="quick-actions">
          <button className="button primary" type="button" onClick={() => onNavigate("inference")}>
            <Play size={17} />
            Run Inference
          </button>
          <button className="button" type="button" onClick={() => onNavigate("train")}>
            <Layers size={17} />
            Start Training
          </button>
          <button className="button" type="button" onClick={() => onNavigate("benchmark")}>
            <Gauge size={17} />
            Benchmark
          </button>
        </div>
      </Card>

      <Card>
        <SectionTitle icon={Terminal} title="Active Training" detail="Live job summary" />
        {activeJob ? (
          <div className="job-summary">
            <div className="job-header">
              <div>
                <strong>{activeJob.jobId}</strong>
                <span>{activeJob.status}</span>
              </div>
              <b>{Math.round(activeJob.progress)}%</b>
            </div>
            <Progress value={activeJob.progress} />
            <div className="metric-row">
              <span>Step {activeJob.step} / {activeJob.maxSteps}</span>
              <span>Loss {activeJob.loss ? formatNumber(activeJob.loss) : "pending"}</span>
              <span>{formatDuration(activeJob.elapsedSeconds)}</span>
            </div>
          </div>
        ) : (
          <EmptyState title="No active training jobs" detail="Training status appears here as soon as a local job starts." />
        )}
      </Card>

      <Card>
        <SectionTitle icon={Activity} title="Recent Activity" detail="Local Core events" />
        <div className="activity-list">
          {activity.map((item) => (
            <div key={item.id} className={`activity-item ${item.tone}`}>
              <span />
              <div>
                <strong>{item.title}</strong>
                <p>{item.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function InferencePage({
  modelOptions,
  selectedModel,
  setSelectedModel,
  prompt,
  setPrompt,
  systemPrompt,
  setSystemPrompt,
  maxTokens,
  setMaxTokens,
  temperature,
  setTemperature,
  topP,
  setTopP,
  inferenceBusy,
  inferenceError,
  inferenceResult,
  onRun,
}: {
  modelOptions: Array<{ value: string; label: string }>;
  selectedModel: string;
  setSelectedModel: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  systemPrompt: string;
  setSystemPrompt: (value: string) => void;
  maxTokens: number;
  setMaxTokens: (value: number) => void;
  temperature: number;
  setTemperature: (value: number) => void;
  topP: number;
  setTopP: (value: number) => void;
  inferenceBusy: boolean;
  inferenceError: string;
  inferenceResult: InferResult | null;
  onRun: () => void;
}) {
  return (
    <div className="page-grid inference-grid">
      <Card>
        <SectionTitle icon={SlidersHorizontal} title="Inference Controls" detail="Local decode settings" />
        <Field label="Model">
          <select
            className="select"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.currentTarget.value)}
          >
            <option value="">Select model</option>
            {modelOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label={`Max tokens: ${maxTokens}`}>
          <input
            className="range"
            type="range"
            min={32}
            max={1024}
            step={16}
            value={maxTokens}
            onChange={(event) => setMaxTokens(event.currentTarget.valueAsNumber)}
          />
        </Field>
        <Field label={`Temperature: ${temperature.toFixed(2)}`}>
          <input
            className="range"
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={temperature}
            onChange={(event) => setTemperature(event.currentTarget.valueAsNumber)}
          />
        </Field>
        <Field label={`Top P: ${topP.toFixed(2)}`}>
          <input
            className="range"
            type="range"
            min={0.1}
            max={1}
            step={0.01}
            value={topP}
            onChange={(event) => setTopP(event.currentTarget.valueAsNumber)}
          />
        </Field>
        <div className="prompt-chips">
          {quickPrompts.map((item) => (
            <button key={item} type="button" onClick={() => setPrompt(item)}>
              {item}
            </button>
          ))}
        </div>
      </Card>

      <Card className="inference-panel">
        <SectionTitle icon={Brain} title="Prompt Workspace" detail="System and user prompt" />
        <Field label="System prompt">
          <textarea
            className="textarea compact"
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.currentTarget.value)}
          />
        </Field>
        <Field label="Prompt">
          <textarea
            className="textarea prompt-input"
            value={prompt}
            onChange={(event) => setPrompt(event.currentTarget.value)}
          />
        </Field>
        {inferenceError && <Notice tone="error">{inferenceError}</Notice>}
        <button className="button primary run-button" type="button" onClick={onRun} disabled={inferenceBusy}>
          {inferenceBusy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          {inferenceBusy ? "Running..." : "Run Local Inference"}
        </button>
      </Card>

      <Card className="output-panel">
        <SectionTitle icon={Terminal} title="Output" detail="Generated response and metrics" />
        {inferenceResult ? (
          <>
            <div className="output-text">{inferenceResult.text}</div>
            <div className="metric-row">
              <span>{inferenceResult.tokensGenerated} tokens</span>
              <span>{formatNumber(inferenceResult.tokensPerSecond)} tok/s</span>
              <span>{inferenceResult.timeMs} ms</span>
            </div>
            <button
              className="button ghost"
              type="button"
              onClick={() => navigator.clipboard?.writeText(inferenceResult.text)}
            >
              <Clipboard size={16} />
              Copy output
            </button>
          </>
        ) : (
          <EmptyState title="No generation yet" detail="Run a prompt to inspect output, latency, and throughput." />
        )}
      </Card>
    </div>
  );
}

function TrainPage({
  config,
  modelOptions,
  jobs,
  busy,
  error,
  onConfigChange,
  onPickDataset,
  onPickOutputDir,
  onStart,
}: {
  config: TrainConfig;
  modelOptions: Array<{ value: string; label: string }>;
  jobs: TrainStatus[];
  busy: boolean;
  error: string;
  onConfigChange: <K extends keyof TrainConfig>(key: K, value: TrainConfig[K]) => void;
  onPickDataset: () => void;
  onPickOutputDir: () => void;
  onStart: () => void;
}) {
  const latest = jobs[0];
  const lossPoints = jobs.filter((job) => typeof job.loss === "number").slice(0, 10).reverse();

  return (
    <div className="page-grid train-grid">
      <Card>
        <SectionTitle icon={Layers} title="Training Setup" detail="LoRA fine tuning profile" />
        <Field label="Base model">
          <select
            className="select"
            value={config.baseModel}
            onChange={(event) => onConfigChange("baseModel", event.currentTarget.value)}
          >
            <option value="">Use active model</option>
            {modelOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Dataset">
          <div className="input-with-action">
            <input className="input" value={config.dataset} readOnly placeholder="Select JSONL, TXT, CSV, or Parquet" />
            <button className="button ghost icon-button" type="button" onClick={onPickDataset}>
              <FolderOpen size={16} />
            </button>
          </div>
        </Field>
        <Field label="Output directory">
          <div className="input-with-action">
            <input className="input" value={config.outputDir} readOnly placeholder="Choose output folder" />
            <button className="button ghost icon-button" type="button" onClick={onPickOutputDir}>
              <FolderOpen size={16} />
            </button>
          </div>
        </Field>
        <div className="two-col">
          <Field label="LoRA rank">
            <input
              className="input"
              type="number"
              min={1}
              value={config.loraRank}
              onChange={(event) => onConfigChange("loraRank", event.currentTarget.valueAsNumber)}
            />
          </Field>
          <Field label="Batch size">
            <input
              className="input"
              type="number"
              min={1}
              value={config.batchSize}
              onChange={(event) => onConfigChange("batchSize", event.currentTarget.valueAsNumber)}
            />
          </Field>
        </div>
        <div className="two-col">
          <Field label="Learning rate">
            <input
              className="input"
              type="number"
              min={0}
              step={0.00001}
              value={config.learningRate}
              onChange={(event) => onConfigChange("learningRate", event.currentTarget.valueAsNumber)}
            />
          </Field>
          <Field label="Max steps">
            <input
              className="input"
              type="number"
              min={1}
              value={config.maxSteps}
              onChange={(event) => onConfigChange("maxSteps", event.currentTarget.valueAsNumber)}
            />
          </Field>
        </div>
        {error && <Notice tone="error">{error}</Notice>}
        <button className="button primary run-button" type="button" onClick={onStart} disabled={busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          {busy ? "Starting..." : "Start Training"}
        </button>
      </Card>

      <Card className="monitor-panel">
        <SectionTitle icon={Activity} title="Training Monitor" detail="Progress, loss, and logs" />
        {latest ? (
          <>
            <div className="job-header large">
              <div>
                <strong>{latest.jobId}</strong>
                <span>{latest.status}</span>
              </div>
              <b>{Math.round(latest.progress)}%</b>
            </div>
            <Progress value={latest.progress} />
            <div className="metric-row">
              <span>Step {latest.step} / {latest.maxSteps}</span>
              <span>Loss {latest.loss ? formatNumber(latest.loss) : "pending"}</span>
              <span>{formatDuration(latest.elapsedSeconds)}</span>
            </div>
            <div className="loss-chart" aria-label="Loss trend">
              {lossPoints.length > 0 ? (
                lossPoints.map((point) => (
                  <span
                    key={`${point.jobId}-${point.step}`}
                    style={{ height: `${Math.max(12, Math.min(96, (point.loss ?? 0) * 28))}%` }}
                    title={`step ${point.step}: ${point.loss}`}
                  />
                ))
              ) : (
                <p>Loss points will appear as Core writes training logs.</p>
              )}
            </div>
            <pre className="log-block">{latest.logTail}</pre>
          </>
        ) : (
          <EmptyState title="No training jobs yet" detail="Start a local fine tuning run to stream progress here." />
        )}
      </Card>
    </div>
  );
}

function BenchmarkPage({
  models,
  selectedModels,
  runs,
  results,
  busy,
  error,
  onToggleModel,
  onRunsChange,
  onRun,
}: {
  models: ModelInfo[];
  selectedModels: string[];
  runs: number;
  results: BenchmarkResult[];
  busy: boolean;
  error: string;
  onToggleModel: (path: string) => void;
  onRunsChange: (runs: number) => void;
  onRun: () => void;
}) {
  const topSpeed = Math.max(...results.map((result) => result.tokensPerSecond), 1);

  return (
    <div className="page-grid benchmark-grid">
      <Card>
        <SectionTitle icon={Gauge} title="Benchmark Setup" detail="Compare local model throughput" />
        <div className="model-check-list">
          {models.length === 0 ? (
            <EmptyState title="No models available" detail="Import a model before running a benchmark." />
          ) : (
            models.map((model) => (
              <label key={model.path} className="check-row">
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.path)}
                  onChange={() => onToggleModel(model.path)}
                />
                <span>
                  <strong>{model.name}</strong>
                  <small>{model.parameters} / {model.quantization} / {formatNumber(model.sizeMb)} MB</small>
                </span>
              </label>
            ))
          )}
        </div>
        <Field label={`Runs: ${runs}`}>
          <input
            className="range"
            type="range"
            min={1}
            max={20}
            value={runs}
            onChange={(event) => onRunsChange(event.currentTarget.valueAsNumber)}
          />
        </Field>
        {error && <Notice tone="error">{error}</Notice>}
        <button className="button primary run-button" type="button" onClick={onRun} disabled={busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <Gauge size={18} />}
          {busy ? "Benchmarking..." : "Run Benchmark"}
        </button>
      </Card>

      <Card className="results-panel">
        <SectionTitle icon={BarChart3} title="Results" detail="Tokens, latency, and memory" />
        {results.length > 0 ? (
          <div className="benchmark-table">
            {results.map((result) => (
              <div key={result.model} className="benchmark-row">
                <div>
                  <strong>{shortName(result.model)}</strong>
                  <small>{result.runs} runs</small>
                </div>
                <div className="bar-track">
                  <span style={{ width: `${(result.tokensPerSecond / topSpeed) * 100}%` }} />
                </div>
                <b>{formatNumber(result.tokensPerSecond)} tok/s</b>
                <span>{formatNumber(result.coldStartMs)} ms</span>
                <span>{formatNumber(result.memoryMb)} MB</span>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState title="No benchmark results" detail="Select one or more models, then run the local benchmark suite." />
        )}
      </Card>
    </div>
  );
}

function ModelsPage({
  models,
  selectedModel,
  modelError,
  downloadUrl,
  downloadBusy,
  onSelect,
  onImport,
  onDownload,
  onDownloadUrlChange,
  onBenchmark,
}: {
  models: ModelInfo[];
  selectedModel: string;
  modelError: string;
  downloadUrl: string;
  downloadBusy: boolean;
  onSelect: (path: string) => void;
  onImport: () => void;
  onDownload: () => void;
  onDownloadUrlChange: (value: string) => void;
  onBenchmark: (path: string) => void;
}) {
  return (
    <div className="page-grid models-grid">
      <Card className="models-panel">
        <SectionTitle icon={Database} title="Local Models" detail="GGUF and GGQR registry" />
        {modelError && <Notice tone="error">{modelError}</Notice>}
        <div className="toolbar">
          <button className="button primary" type="button" onClick={onImport}>
            <Upload size={16} />
            Import Model
          </button>
          <div className="download-box">
            <input
              className="input"
              value={downloadUrl}
              onChange={(event) => onDownloadUrlChange(event.currentTarget.value)}
              placeholder="Model URL"
            />
            <button className="button" type="button" onClick={onDownload} disabled={downloadBusy}>
              {downloadBusy ? <Loader2 className="spin" size={16} /> : <Download size={16} />}
              Download
            </button>
          </div>
        </div>

        <div className="model-list">
          {models.length === 0 ? (
            <EmptyState title="No local models found" detail="Import a GGUF or GGQR model to populate the desktop registry." />
          ) : (
            models.map((model) => (
              <article key={model.path} className={`model-card ${selectedModel === model.path ? "selected" : ""}`}>
                <div className="model-main">
                  <div className="model-icon">
                    <Brain size={20} />
                  </div>
                  <div>
                    <h3>{model.name}</h3>
                    <p>{model.path}</p>
                  </div>
                </div>
                <div className="model-meta">
                  <span>{model.parameters}</span>
                  <span>{model.quantization}</span>
                  <span>{formatNumber(model.sizeMb)} MB</span>
                  <span>{formatDate(model.modifiedMs)}</span>
                </div>
                <div className="model-actions">
                  <button className="button ghost" type="button" onClick={() => onSelect(model.path)}>
                    <CheckCircle2 size={16} />
                    Load
                  </button>
                  <button className="button ghost" type="button" onClick={() => onBenchmark(model.path)}>
                    <Gauge size={16} />
                    Benchmark
                  </button>
                  <button className="button ghost" type="button" onClick={() => navigator.clipboard?.writeText(model.path)}>
                    <Clipboard size={16} />
                    Copy Path
                  </button>
                </div>
              </article>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}

function SettingsPage({ stats }: { stats: SystemStats }) {
  return (
    <div className="page-grid settings-grid">
      <Card>
        <SectionTitle icon={Settings} title="Runtime Settings" detail="Local paths and Core defaults" />
        <Field label="GwenLand binary">
          <input className="input mono" value={stats.binaryPath} readOnly />
        </Field>
        <Field label="Models directory">
          <input className="input mono" value={stats.modelsDir} readOnly />
        </Field>
        <div className="two-col">
          <Field label="Default max tokens">
            <input className="input" type="number" defaultValue={160} />
          </Field>
          <Field label="Default temperature">
            <input className="input" type="number" step={0.05} defaultValue={0.7} />
          </Field>
        </div>
      </Card>

      <Card>
        <SectionTitle icon={HardDrive} title="About This Build" detail="Standalone desktop app" />
        <div className="about-stack">
          <InfoRow label="Frontend" value="Tauri desktop UI" />
          <InfoRow label="Backend" value="GwenLand Core subprocess commands" />
          <InfoRow label="Cloud services" value="None" />
          <InfoRow label="Theme" value="Warm dark / orange Core palette" />
          <InfoRow label="Storage" value="~/.gwenland local workspace" />
        </div>
      </Card>
    </div>
  );
}

function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{children}</section>;
}

function StatCard({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: IconType;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <Card className="stat-card">
      <Icon size={20} />
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{hint}</p>
    </Card>
  );
}

function SectionTitle({ icon: Icon, title, detail }: { icon: IconType; title: string; detail: string }) {
  return (
    <div className="section-title">
      <Icon size={18} />
      <div>
        <h2>{title}</h2>
        <p>{detail}</p>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function Notice({ tone, children }: { tone: "error" | "info"; children: ReactNode }) {
  return <div className={`notice ${tone}`}>{children}</div>;
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <Terminal size={20} />
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}

function Progress({ value }: { value: number }) {
  return (
    <div className="progress">
      <span style={{ width: `${Math.max(0, Math.min(value, 100))}%` }} />
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function titleForPage(page: PageId) {
  switch (page) {
    case "dashboard":
      return "Dashboard";
    case "inference":
      return "Inference";
    case "train":
      return "Train";
    case "benchmark":
      return "Benchmark";
    case "models":
      return "Models";
    case "settings":
      return "Settings";
    default:
      return "Dashboard";
  }
}

function readError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function shortName(path: string) {
  return path.split(/[\\/]/).pop() || path || "model";
}

function shortPath(path: string) {
  if (!path) return "output";
  const parts = path.split(/[\\/]/);
  return parts.slice(-2).join("/");
}

function formatNumber(value: number) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value || 0);
}

function formatDuration(seconds: number) {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder}s`;
}

function formatDate(ms: number) {
  if (!ms) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(ms));
}

export default App;
