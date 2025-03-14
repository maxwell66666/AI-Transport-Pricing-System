import { useState } from 'react'
import { Button } from '@/components/ui/Button'
import { useQuote } from '@/lib/hooks/useQuote'
import { QuoteRequest } from '@/lib/api/client'

export function QuoteForm() {
  const { loading, error, quote, requestQuote } = useQuote()
  const [formData, setFormData] = useState<QuoteRequest>({
    origin: '',
    destination: '',
    weight: 0,
    volume: 0,
    goodsType: '',
    serviceLevel: 'standard',
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'weight' || name === 'volume' ? parseFloat(value) || 0 : value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await requestQuote(formData)
    } catch (err) {
      console.error('报价请求失败:', err)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div>
          <label htmlFor="origin" className="block text-sm font-medium text-gray-700">
            起始地
          </label>
          <input
            type="text"
            id="origin"
            name="origin"
            value={formData.origin}
            onChange={handleChange}
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          />
        </div>

        <div>
          <label htmlFor="destination" className="block text-sm font-medium text-gray-700">
            目的地
          </label>
          <input
            type="text"
            id="destination"
            name="destination"
            value={formData.destination}
            onChange={handleChange}
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          />
        </div>

        <div>
          <label htmlFor="weight" className="block text-sm font-medium text-gray-700">
            重量 (kg)
          </label>
          <input
            type="number"
            id="weight"
            name="weight"
            value={formData.weight}
            onChange={handleChange}
            required
            min="0"
            step="0.1"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          />
        </div>

        <div>
          <label htmlFor="volume" className="block text-sm font-medium text-gray-700">
            体积 (m³)
          </label>
          <input
            type="number"
            id="volume"
            name="volume"
            value={formData.volume}
            onChange={handleChange}
            required
            min="0"
            step="0.01"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          />
        </div>

        <div>
          <label htmlFor="goodsType" className="block text-sm font-medium text-gray-700">
            货物类型
          </label>
          <select
            id="goodsType"
            name="goodsType"
            value={formData.goodsType}
            onChange={handleChange}
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          >
            <option value="">请选择货物类型</option>
            <option value="general">普通货物</option>
            <option value="fragile">易碎品</option>
            <option value="dangerous">危险品</option>
            <option value="perishable">易腐品</option>
          </select>
        </div>

        <div>
          <label htmlFor="serviceLevel" className="block text-sm font-medium text-gray-700">
            服务等级
          </label>
          <select
            id="serviceLevel"
            name="serviceLevel"
            value={formData.serviceLevel}
            onChange={handleChange}
            required
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          >
            <option value="economy">经济</option>
            <option value="standard">标准</option>
            <option value="express">快速</option>
            <option value="premium">优质</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">错误</h3>
              <div className="mt-2 text-sm text-red-700">{error}</div>
            </div>
          </div>
        </div>
      )}

      <Button type="submit" disabled={loading}>
        {loading ? '获取报价中...' : '获取报价'}
      </Button>

      {quote && (
        <div className="mt-6 rounded-lg bg-white p-6 shadow">
          <h3 className="text-lg font-medium leading-6 text-gray-900">报价结果</h3>
          <dl className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2">
            <div className="overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:p-6">
              <dt className="truncate text-sm font-medium text-gray-500">报价金额</dt>
              <dd className="mt-1 text-3xl font-semibold tracking-tight text-gray-900">
                ¥{quote.price.toFixed(2)}
              </dd>
            </div>
            <div className="overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:p-6">
              <dt className="truncate text-sm font-medium text-gray-500">预计送达时间</dt>
              <dd className="mt-1 text-3xl font-semibold tracking-tight text-gray-900">
                {quote.estimatedTime}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </form>
  )
} 