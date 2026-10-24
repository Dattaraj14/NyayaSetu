@echo off
set PATH=%PATH%;C:\Program Files\nodejs
rmdir /s /q frontend
npx -y create-vite@latest frontend --template react-ts
cd frontend
npm install
npm install tailwindcss postcss autoprefixer react-router-dom axios lucide-react
npx tailwindcss init -p
