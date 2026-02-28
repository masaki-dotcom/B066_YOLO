import { defineStore } from 'pinia'

export const UrlStore = defineStore('url_name', {
  state: () => ({
    //  smart_mat_url:"http://176.72.74.148:5001/api/smart_mat_url/", 

     //---------------------------------------------------------------

     smart_mat_url:"http://localhost:5001/api/smart_mat_url/", 
  }),
  
})