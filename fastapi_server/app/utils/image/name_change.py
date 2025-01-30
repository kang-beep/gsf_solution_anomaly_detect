import os

def rename_files(folder_path):
   # 폴더가 존재하는지 확인
   if not os.path.exists(folder_path):
       print(f"Error: 폴더를 찾을 수 없습니다: {folder_path}")
       return False

   files = os.listdir(folder_path)
   png_files = [f for f in files if f.endswith('.png')]
   
   if not png_files:
       print("해당 폴더에 PNG 파일이 없습니다.")
       return False
       
   folder_name = os.path.basename(folder_path)
   
   # 각 파일 이름 변경
   for file_name in png_files:
       new_name = f"{folder_name}_{file_name}"
       old_path = os.path.join(folder_path, file_name)
       new_path = os.path.join(folder_path, new_name)
       
       os.rename(old_path, new_path)
       print(f"Renamed: {file_name} -> {new_name}")
   
   print(f"\n총 {len(png_files)}개의 파일 이름이 변경되었습니다.")
   return True

def main():
   while True:
       # 폴더 경로 입력 받기
       folder_path = input("\n폴더 경로를 입력하세요: ")
       
       # 파일 이름 변경 실행
       success = rename_files(folder_path)
       
       while True:
           if not success:
               response = input("\n다른 폴더를 시도하시겠습니까? (C: 계속, Q: 종료): ").upper()
           else:
               response = input("\n다른 폴더의 파일 이름을 변경하시겠습니까? (C: 계속, Q: 종료): ").upper()
               
           if response in ['C', 'Q']:
               break
           print("잘못된 입력입니다. C 또는 Q를 입력해주세요.")
       
       if response == 'Q':
           print("프로그램을 종료합니다.")
           break

if __name__ == "__main__":
   main()