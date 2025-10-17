import React from 'react'
import { Wrench, ArrowRight } from 'lucide-react'

interface Props {
  eda?: any
  modeling?: any
}

export function FeatureEngineeringView({ eda, modeling }: Props) {
  const originalColumns = eda?.columns || []
  const numericCols = originalColumns.filter((c: any) => c.type === 'numeric')
  const categoricalCols = originalColumns.filter((c: any) => c.type === 'categorical')
  
  const features = modeling?.features || []
  const selectedTools = modeling?.selected_tools || []

  return (
    <div className="card">
      <h3 className="flex items-center gap-2 mb-4">
        <Wrench size={20} />
        Feature Engineering
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="p-4 bg-surface rounded">
          <h4 className="text-sm font-semibold mb-2">Original Dataset</h4>
          <div className="text-2xl font-bold">{originalColumns.length}</div>
          <div className="text-sm text-muted">
            {numericCols.length} numeric, {categoricalCols.length} categorical
          </div>
        </div>

        <div className="p-4 bg-surface rounded">
          <h4 className="text-sm font-semibold mb-2">After Preprocessing</h4>
          <div className="text-2xl font-bold">{features.length || originalColumns.length}</div>
          <div className="text-sm text-muted">
            features used in modeling
          </div>
        </div>
      </div>

      <div className="mb-4">
        <h4 className="text-sm font-semibold mb-2">Transformations Applied</h4>
        <div className="space-y-2">
          {numericCols.length > 0 && (
            <div className="flex items-center gap-2 p-2 bg-surface rounded">
              <div className="badge">Numeric</div>
              <ArrowRight size={16} className="text-muted" />
              <span className="text-sm">StandardScaler (normalization)</span>
            </div>
          )}
          
          {categoricalCols.length > 0 && (
            <div className="flex items-center gap-2 p-2 bg-surface rounded">
              <div className="badge">Categorical</div>
              <ArrowRight size={16} className="text-muted" />
              <span className="text-sm">OneHotEncoder / TopK Encoding</span>
            </div>
          )}

          {eda?.missing_summary && (
            <div className="flex items-center gap-2 p-2 bg-surface rounded">
              <div className="badge">Missing Values</div>
              <ArrowRight size={16} className="text-muted" />
              <span className="text-sm">SimpleImputer (median/mode)</span>
            </div>
          )}
        </div>
      </div>

      {selectedTools.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-2">Selected Tools</h4>
          <div className="flex flex-wrap gap-2">
            {selectedTools.map((tool: string, idx: number) => (
              <span key={idx} className="badge badge-primary">
                {tool}
              </span>
            ))}
          </div>
        </div>
      )}

      {features.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-semibold mb-2">Feature List</h4>
          <div className="max-h-48 overflow-y-auto">
            <div className="flex flex-wrap gap-1">
              {features.map((feature: string, idx: number) => (
                <span key={idx} className="text-xs px-2 py-1 bg-surface rounded">
                  {feature}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

