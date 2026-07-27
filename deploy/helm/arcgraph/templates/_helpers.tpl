{{/*
W25-OPS-PROD / ADR-093-amendment-01 §D-4 — Helm template helpers.
*/}}

{{- define "arcgraph.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "arcgraph.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "arcgraph.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "arcgraph.labels" -}}
helm.sh/chart: {{ include "arcgraph.chart" . }}
{{ include "arcgraph.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "arcgraph.selectorLabels" -}}
app.kubernetes.io/name: {{ include "arcgraph.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "arcgraph.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "arcgraph.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Admin port from the bind string. Used by Service + NetworkPolicy +
probe definitions to stay in sync with values.admin.bind.

Supports both IPv4 (`0.0.0.0:8090`, `127.0.0.1:8090`) and IPv6
(`[::1]:8090`, `[::]:8090`) forms by extracting the trailing
`:<port>` via regex instead of splitting on `:` (which would break
on IPv6 since `[::1]:8090` has multiple colons).
*/}}
{{- define "arcgraph.adminPort" -}}
{{- $bind := .Values.admin.bind -}}
{{- $port := regexFind ":[0-9]+$" $bind -}}
{{- if $port -}}
{{- $port | trimPrefix ":" -}}
{{- else -}}
{{- fail (printf "arcgraph.adminPort: could not extract :<port> from admin.bind=%q (expected forms: HOST:PORT or [IPv6]:PORT)" $bind) -}}
{{- end -}}
{{- end -}}
