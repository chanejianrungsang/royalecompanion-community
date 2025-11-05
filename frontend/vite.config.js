import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, mkdirSync, readdirSync, statSync } from 'fs'
import { join } from 'path'

// Plugin to copy assets during build
function copyAssetsPlugin() {
  return {
    name: 'copy-assets',
    closeBundle() {
      const srcDir = join(__dirname, '..', 'assets', 'card_images')
      const destDir = join(__dirname, 'dist', 'card_images')
      
      function copyRecursive(src, dest) {
        mkdirSync(dest, { recursive: true })
        const entries = readdirSync(src)
        
        for (const entry of entries) {
          const srcPath = join(src, entry)
          const destPath = join(dest, entry)
          
          if (statSync(srcPath).isDirectory()) {
            copyRecursive(srcPath, destPath)
          } else {
            copyFileSync(srcPath, destPath)
          }
        }
      }
      
      console.log('Copying card images to dist...')
      copyRecursive(srcDir, destDir)
      console.log('Card images copied successfully!')
    }
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), copyAssetsPlugin()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true
  },
  server: {
    port: 5173,
    strictPort: true
  }
})


