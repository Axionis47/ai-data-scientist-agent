import React from 'react'
import { Activity, TrendingUp, BarChart2, AlertCircle, CheckCircle } from 'lucide-react'

interface Props {
  analytics?: {
    treatment?: string
    outcome?: string
    confounders?: string[]
    effect_estimate?: {
      method?: string
      effect?: number
      ci_lower?: number
      ci_upper?: number
      p_value?: number
      interpretation?: string
    }
    double_ml_estimate?: {
      method?: string
      effect?: number
      ci_lower?: number
      ci_upper?: number
      p_value?: number
      is_significant?: boolean
      interpretation?: string
    }
    matching_estimate?: {
      method?: string
      att?: number
      n_treated?: number
      n_matched?: number
    }
    assumptions?: Array<{
      name?: string
      description?: string
      status?: string
    }>
    recommendations?: string[]
    // Time series
    stationarity?: {
      adf_pvalue?: number
      is_stationary?: boolean
    }
    forecast?: {
      method?: string
      predictions?: number[]
    }
    // Statistical
    normality?: {
      test?: string
      p_value?: number
      is_normal?: boolean
    }
    correlations?: Record<string, number>
  }
  analysisType?: string
}

export function AdvancedAnalyticsView({ analytics, analysisType }: Props) {
  if (!analytics || Object.keys(analytics).length === 0) {
    return null
  }

  const isCausal = analysisType === 'causal' || analytics.effect_estimate || analytics.double_ml_estimate
  const isTimeSeries = analysisType === 'time_series' || analytics.stationarity || analytics.forecast
  const isStatistical = analysisType === 'statistical' || analytics.normality

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <Activity size={20} />
        Advanced Analytics
        {isCausal && <span className="badge ml-2">Causal Inference</span>}
        {isTimeSeries && <span className="badge ml-2">Time Series</span>}
        {isStatistical && <span className="badge ml-2">Statistical</span>}
      </h3>

      {/* Causal Inference Results */}
      {isCausal && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <TrendingUp size={16} />
            Treatment Effect Analysis
          </h4>
          
          {analytics.treatment && analytics.outcome && (
            <div className="p-3 bg-surface rounded mb-3">
              <div className="text-sm">
                <strong>Treatment:</strong> {analytics.treatment} → <strong>Outcome:</strong> {analytics.outcome}
              </div>
              {analytics.confounders && analytics.confounders.length > 0 && (
                <div className="text-sm text-muted mt-1">
                  <strong>Confounders:</strong> {analytics.confounders.join(', ')}
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Primary Effect Estimate */}
            {analytics.effect_estimate && (
              <div className="p-4 bg-surface rounded">
                <div className="text-xs text-muted mb-1">{analytics.effect_estimate.method || 'DoWhy'}</div>
                <div className="text-2xl font-bold">
                  {analytics.effect_estimate.effect?.toFixed(4) || 'N/A'}
                </div>
                {analytics.effect_estimate.ci_lower !== undefined && (
                  <div className="text-sm text-muted">
                    95% CI: [{analytics.effect_estimate.ci_lower?.toFixed(4)}, {analytics.effect_estimate.ci_upper?.toFixed(4)}]
                  </div>
                )}
                {analytics.effect_estimate.p_value !== undefined && (
                  <div className="text-sm">
                    p-value: {analytics.effect_estimate.p_value?.toFixed(4)}
                    {analytics.effect_estimate.p_value < 0.05 ? (
                      <CheckCircle size={14} className="inline ml-1 text-success" />
                    ) : (
                      <AlertCircle size={14} className="inline ml-1 text-warning" />
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Double ML Estimate */}
            {analytics.double_ml_estimate && (
              <div className="p-4 bg-surface rounded border-l-4 border-purple-500">
                <div className="text-xs text-muted mb-1">{analytics.double_ml_estimate.method || 'Double ML'}</div>
                <div className="text-2xl font-bold">
                  {analytics.double_ml_estimate.effect?.toFixed(4) || 'N/A'}
                </div>
                {analytics.double_ml_estimate.ci_lower !== undefined && (
                  <div className="text-sm text-muted">
                    95% CI: [{analytics.double_ml_estimate.ci_lower?.toFixed(4)}, {analytics.double_ml_estimate.ci_upper?.toFixed(4)}]
                  </div>
                )}
                <div className="text-sm">
                  {analytics.double_ml_estimate.is_significant ? (
                    <span className="text-success flex items-center gap-1">
                      <CheckCircle size={14} /> Statistically Significant
                    </span>
                  ) : (
                    <span className="text-warning flex items-center gap-1">
                      <AlertCircle size={14} /> Not Significant
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Interpretation */}
          {(analytics.effect_estimate?.interpretation || analytics.double_ml_estimate?.interpretation) && (
            <div className="alert alert-info mt-4">
              <p className="m-0 text-sm">
                {analytics.double_ml_estimate?.interpretation || analytics.effect_estimate?.interpretation}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Recommendations */}
      {analytics.recommendations && analytics.recommendations.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold mb-2">Recommendations</h4>
          <ul className="text-sm space-y-1">
            {analytics.recommendations.map((rec, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <BarChart2 size={14} className="mt-1 flex-shrink-0" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

