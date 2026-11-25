import { Monitor, Moon, RotateCcw, Sun } from "lucide-react";
import { eps } from "@/lib/format";
import { useInfo } from "@/lib/queries";
import { useSettings, type Theme } from "@/lib/settings";
import { Button, Card, CardBody, CardHeader, Field, Input, PageHeader, Segmented } from "@/components/ui/primitives";
import { Select, Switch } from "@/components/ui/radix";
import { KV } from "@/components/domain";

export default function SettingsPage() {
  const { settings, update, reset } = useSettings();
  const info = useInfo();
  const epsilons = info.data?.epsilons ?? [0.2, 0.1, 0.05, 0.02, 0.01];
  return (
    <>
      <PageHeader
        title="Settings"
        description="Preferences are stored in this browser only. They change what the console shows and requests; they never change a calibrated threshold."
        actions={<Button variant="outline" size="sm" onClick={reset}><RotateCcw className="h-3.5 w-3.5" /> Reset to defaults</Button>}
      />
      <div className="grid max-w-4xl gap-4">
        <Card>
          <CardHeader title="Appearance" />
          <CardBody>
            <Segmented<Theme>
              ariaLabel="Theme"
              value={settings.theme}
              onChange={(theme) => update({ theme })}
              options={[
                { value: "light", label: <span className="flex items-center gap-1.5"><Sun className="h-3.5 w-3.5" /> Light</span> },
                { value: "dark", label: <span className="flex items-center gap-1.5"><Moon className="h-3.5 w-3.5" /> Dark</span> },
                { value: "system", label: <span className="flex items-center gap-1.5"><Monitor className="h-3.5 w-3.5" /> System</span> },
              ]}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Triage defaults" />
          <CardBody className="grid gap-5 md:grid-cols-2">
            <Field label="Default ε" hint="Used for new inspections, the batch console and the overview." htmlFor="def-eps">
              <Select id="def-eps" value={String(settings.epsilon)} onValueChange={(v) => update({ epsilon: Number(v) })} options={epsilons.map((e) => ({ value: String(e), label: eps(e) }))} />
            </Field>
            <Field label="Batch size" hint="Images per batch in the batch console (1–96). Each is inspected once." htmlFor="bs">
              <Input id="bs" type="number" min={1} max={96} value={settings.batchSize} onChange={(e) => update({ batchSize: Math.min(96, Math.max(1, Math.floor(Number(e.target.value) || 1))) })} className="num" />
            </Field>
            <Field label="Images per day" hint="For the projected review hours." htmlFor="ipd">
              <Input id="ipd" type="number" min={1} step={100} value={settings.imagesPerDay} onChange={(e) => update({ imagesPerDay: Math.max(1, Number(e.target.value) || 1) })} className="num" />
            </Field>
            <Field label="Minutes per review" htmlFor="mpr2">
              <Input id="mpr2" type="number" min={0.1} step={0.1} value={settings.minutesPerReview} onChange={(e) => update({ minutesPerReview: Math.max(0.1, Number(e.target.value) || 0.1) })} className="num" />
            </Field>
            <Field label="Parallel requests" hint="Concurrent /v1/inspect calls while a batch is inspected (1–8)." htmlFor="cc">
              <Input id="cc" type="number" min={1} max={8} value={settings.concurrency} onChange={(e) => update({ concurrency: Math.min(8, Math.max(1, Math.floor(Number(e.target.value) || 1))) })} className="num" />
            </Field>
            <div className="flex items-center justify-between gap-4 rounded-lg border border-border p-3">
              <div>
                <label htmlFor="truth" className="text-[13px] font-medium">Show ground truth</label>
                <p className="text-[12px] text-subtle">Red borders and miss counts in the batch console.</p>
              </div>
              <Switch id="truth" checked={settings.showTruth} onCheckedChange={(v) => update({ showTruth: v })} />
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Connection" description="The console talks to the service on the same origin; in development Vite proxies /v1 to port 8000." />
          <CardBody className="pt-3">
            <dl className="divide-y divide-border">
              <KV k="Status" v={info.isSuccess ? "online" : info.isPending ? "connecting" : "offline"} mono={false} />
              <KV k="Service" v={info.data ? `${info.data.service} ${info.data.version}` : "—"} mono={false} />
              <KV k="Sample images" v={info.data ? (info.data.samples_available ? info.data.sample_dataset : "not available") : "—"} />
            </dl>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
