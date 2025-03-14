import { QuoteForm } from '@/components/forms/QuoteForm'

export default function QuotePage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mx-auto max-w-3xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">获取运输报价</h1>
          <p className="mt-2 text-gray-600">
            请填写以下表单获取运输报价。我们的AI系统会为您计算最优惠的价格。
          </p>
        </div>
        <div className="rounded-lg bg-white p-6 shadow">
          <QuoteForm />
        </div>
      </div>
    </div>
  )
} 