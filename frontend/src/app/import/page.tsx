import { useState } from 'react'
import { Button } from '@/components/ui/Button'

export default function ImportPage() {
  const [files, setFiles] = useState<FileList | null>(null)
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState<any[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(e.target.files)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      setError('请选择要上传的文件')
      return
    }

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      Array.from(files).forEach((file) => {
        formData.append('files', file)
      })

      const response = await fetch('/api/quotes/import', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('上传失败')
      }

      const data = await response.json()
      setResults(data.results)
    } catch (err) {
      setError(err instanceof Error ? err.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mx-auto max-w-3xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">导入历史报价</h1>
          <p className="mt-2 text-gray-600">
            支持上传PDF格式的报价单和邮件文件（.eml或.msg格式）
          </p>
        </div>

        <div className="rounded-lg bg-white p-6 shadow">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700">选择文件</label>
            <div className="mt-2">
              <input
                type="file"
                multiple
                accept=".pdf,.eml,.msg"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-primary file:text-white
                  hover:file:bg-primary/90"
              />
            </div>
            <p className="mt-2 text-sm text-gray-500">
              可以同时选择多个文件进行上传
            </p>
          </div>

          <Button
            onClick={handleUpload}
            disabled={uploading || !files}
            className="w-full"
          >
            {uploading ? '处理中...' : '开始导入'}
          </Button>

          {error && (
            <div className="mt-4 rounded-md bg-red-50 p-4">
              <div className="flex">
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">错误</h3>
                  <div className="mt-2 text-sm text-red-700">{error}</div>
                </div>
              </div>
            </div>
          )}

          {results.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-medium text-gray-900">处理结果</h3>
              <div className="mt-4">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-300">
                    <thead>
                      <tr>
                        <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                          文件名
                        </th>
                        <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                          状态
                        </th>
                        <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                          提取信息
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {results.map((result, index) => (
                        <tr key={index}>
                          <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                            {result.filename}
                          </td>
                          <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                            {result.success ? (
                              <span className="text-green-600">成功</span>
                            ) : (
                              <span className="text-red-600">失败</span>
                            )}
                          </td>
                          <td className="px-3 py-4 text-sm text-gray-500">
                            {result.success ? (
                              <pre className="whitespace-pre-wrap">
                                {JSON.stringify(result.data, null, 2)}
                              </pre>
                            ) : (
                              result.error
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 