@echo off
set PATH=%PATH%;C:\Program Files\nodejs
cd frontend
call npm install
call npm install tailwindcss postcss autoprefixer react-router-dom axios lucide-react
call npx tailwindcss init -p
