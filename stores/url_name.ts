import { defineStore } from 'pinia'

export const UrlStore = defineStore('url_name', {
  state: () => ({
    //  yolo_server_url:"http://176.72.74.148:5005/api/yolo_server_url/", 

     //---------------------------------------------------------------

     yolo_server_url:"http://localhost:5005/api/yolo_server_url/", 
  }),
  
})