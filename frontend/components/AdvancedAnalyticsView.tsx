import React from 'react'
import { Activity, TrendingUp, BarChart2, AlertCircle, CheckCircle, Clock, LineChart } from 'lucide-react'

interface Props {
  analytics?: {
    // Causal inference
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
      robust?: boolean
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
      testable?: boolean
    }>
    recommendations?: string[]
    // Time series
    stationarity?: {
      adf?: { p_value?: number; reject_null?: boolean }
      kpss?: { p_value?: number; reject_null?: boolean }
      is_stationary?: boolean
      interpretation?: string
    }
    decomposition?: {
      period?: number
      trend_strength?: number
      seasonal_strength?: number
      interpretation?: string
    }
    frequency?: {
      frequency?: string
      interpretation?: string
    }
    forecast?: {
      method?: string
      order?: number[]
      forecast?: number[]
      confidence_lower?: number[]
      confidence_upper?: number[]
      horizon?: number
      interpretation?: string
    }
    // Statistical
    normality_tests?: Array<{
      column?: string
      test_name?: string
      p_value?: number
      is_normal?: boolean
      interpretation?: string
    }>
    correlation_tests?: Array<{
      column?: string
      correlation?: number
      p_value?: number
      is_significant?: boolean
      strength?: string
      direction?: string
    }>
    group_tests?: Array<{
      group_col?: string
      test_name?: string
      p_value?: number
      is_significant?: boolean
      interpretation?: string
    }>
    summary?: Record<string, unknown>
  }
  analysisType?: string
}

export function AdvancedAnalyticsView({ analytics, analysisType }: Props) {
  if (!analytics || Object.keys(analytics).length === 0) {
    return null
  }

  const isCausal = analysisType === 'causal' || analytics.effect_estimate || analytics.double_ml_estimate
  const isTimeSeries = analysisType === 'time_series' || analytics.stationarity || analytics.forecast || analytics.decomposition
  const isStatistical = analysisType === 'statistical' || analytics.normality_tests || analytics.correlation_tests

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

      {/* Time Series Results */}
      {isTimeSeries && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Clock size={16} />
            Time Series Analysis
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {/* Frequency Detection */}
            {analytics.frequency && (
              <div className="p-4 bg-surface rounded">
                <div className="text-xs text-muted mb-1">Detected Frequency</div>
                <div className="text-xl font-bold">{analytics.frequency.interpretation || analytics.frequency.frequency || 'Unknown'}</div>
              </div>
            )}

            {/* Stationarity */}
            {analytics.stationarity && (
              <div className="p-4 bg-surface rounded">
                <div className="text-xs text-muted mb-1">Stationarity</div>
                <div className="flex items-center gap-2">
                  {analytics.stationarity.is_stationary ? (
                    <>
                      <CheckCircle size={18} className="text-success" />
                      <span className="text-lg font-semibold text-success">Stationary</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle size={18} className="text-warning" />
                      <span className="text-lg font-semibold text-warning">Non-Stationary</span>
                    </>
                  )}
                </div>
                {analytics.stationarity.adf?.p_value !== undefined && (
                  <div className="text-xs text-muted mt-1">ADF p={analytics.stationarity.adf.p_value.toFixed(4)}</div>
                )}
              </div>
            )}

            {/* Decomposition */}
            {analytics.decomposition && (
              <div className="p-4 bg-surface rounded">
                <div className="text-xs text-muted mb-1">Decomposition (period={analytics.decomposition.period})</div>
                <div className="text-sm">
                  <div>Trend: <strong>{((analytics.decomposition.trend_strength || 0) * 100).toFixed(0)}%</strong></div>
                  <div>Seasonal: <strong>{((analytics.decomposition.seasonal_strength || 0) * 100).toFixed(0)}%</strong></div>
                </div>
              </div>
            )}
          </div>

          {/* Forecast */}
          {analytics.forecast && analytics.forecast.forecast && (
            <div className="p-4 bg-surface rounded">
              <div className="text-xs text-muted mb-2">
                {analytics.forecast.method || 'ARIMA'} Forecast ({analytics.forecast.horizon} periods)
              </div>
              <div className="flex flex-wrap gap-2">
                {analytics.forecast.forecast.slice(0, 10).map((val, idx) => (
                  <div key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900 rounded text-sm">
                    <span className="text-xs text-muted">t+{idx + 1}:</span> {val.toFixed(2)}
                  </div>
                ))}
                {analytics.forecast.forecast.length > 10 && (
                  <div className="px-2 py-1 text-muted text-sm">+{analytics.forecast.forecast.length - 10} more</div>
                )}
              </div>
              {analytics.forecast.interpretation && (
                <div className="text-xs text-muted mt-2">{analytics.forecast.interpretation}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Statistical Testing Results */}
      {isStatistical && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <LineChart size={16} />
            Statistical Tests
          </h4>

          {/* Normality Tests */}
          {analytics.normality_tests && analytics.normality_tests.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-muted mb-2">Normality Tests</div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Column</th>
                      <th className="text-left py-2 px-2">Test</th>
                      <th className="text-right py-2 px-2">p-value</th>
                      <th className="text-center py-2 px-2">Normal?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.normality_tests.slice(0, 10).map((test, idx) => (
                      <tr key={idx} className="border-b border-gray-100 dark:border-gray-800">
                        <td className="py-2 px-2 font-mono text-xs">{test.column}</td>
                        <td className="py-2 px-2">{test.test_name}</td>
                        <td className="py-2 px-2 text-right">{test.p_value?.toFixed(4) || 'N/A'}</td>
                        <td className="py-2 px-2 text-center">
                          {test.is_normal ? (
                            <CheckCircle size={14} className="inline text-success" />
                          ) : (
                            <AlertCircle size={14} className="inline text-warning" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Correlation Tests */}
          {analytics.correlation_tests && analytics.correlation_tests.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-muted mb-2">Significant Correlations</div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {analytics.correlation_tests.filter(t => t.is_significant).slice(0, 9).map((test, idx) => (
                  <div key={idx} className="p-3 bg-surface rounded flex justify-between items-center">
                    <span className="font-mono text-xs">{test.column}</span>
                    <span className={`font-bold ${(test.correlation || 0) > 0 ? 'text-success' : 'text-error'}`}>
                      {test.correlation?.toFixed(3)} ({test.strength})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Group Tests */}
          {analytics.group_tests && analytics.group_tests.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-muted mb-2">Group Difference Tests</div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Group Variable</th>
                      <th className="text-left py-2 px-2">Test</th>
                      <th className="text-right py-2 px-2">p-value</th>
                      <th className="text-center py-2 px-2">Significant?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.group_tests.slice(0, 10).map((test, idx) => (
                      <tr key={idx} className="border-b border-gray-100 dark:border-gray-800">
                        <td className="py-2 px-2 font-mono text-xs">{test.group_col}</td>
                        <td className="py-2 px-2">{test.test_name}</td>
                        <td className="py-2 px-2 text-right">{test.p_value?.toFixed(4) || 'N/A'}</td>
                        <td className="py-2 px-2 text-center">
                          {test.is_significant ? (
                            <CheckCircle size={14} className="inline text-success" />
                          ) : (
                            <AlertCircle size={14} className="inline text-muted" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
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

