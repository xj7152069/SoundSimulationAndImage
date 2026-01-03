#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <condition_variable>
#include "../xjc.h"
class ThreadPool{
private:
    std::vector<std::thread> threads;
    std::queue <std::function<void()>> tasks;
//mutex对象用于对语句块上锁，
//对于调用了mutex对象的线程，同一时间只能有一个线程对该语句块进行操作；
//没有调用mutex对象的线程其实可以直接调用语句块中的变量，但这会破坏其他线程的互斥性，并导致数据混乱；
    std::mutex mtx,mtxAddTask,mtxCountFinishedTask;
    std::condition_variable condition,conditionAdd;
    bool stop;
    int maxTask,maxThread;
    int AddTask,FinishedTask;

public:
    int WaitForTask(){
        while(AddTask!=FinishedTask){
            std::unique_lock<std::mutex>lock(mtxAddTask);
            if(!tasks.empty())condition.notify_one();
        }
        return FinishedTask;
    }
    void NotifyAllThread(){condition.notify_all();}
    int GetTaskNum(){return tasks.size();}
    void ClearTaskCount(){AddTask=0,FinishedTask=0;}
    ThreadPool(int numThreads,int numTask)\
        :stop(false),maxTask(2),maxThread(1),AddTask(0),FinishedTask(0)\
    {
        numThreads=max(numThreads,1);
        maxThread=numThreads+1;
        maxTask=max(numTask,maxThread);

        //调用构造函数的时候就启动了所有线程；
        for(int i=0;i<numThreads;i++){
            //用lamda表达式写了循环函数输入线程，使其一直接收任务；
            threads.emplace_back([this]{
                while(1){
                    std::unique_lock<std::mutex>lock(mtx);
                    condition.wait(lock, [this] {
                        return !tasks.empty() || stop;
                    });
                    if(stop && tasks.empty())return;
                    std::function<void()> task(std::move(tasks.front()));
                    tasks.pop();
                    conditionAdd.notify_one();
                    lock.unlock();  //解锁，允许其他线程操作 任务队列

                    task();
                    {
                        std::unique_lock<std::mutex>\
                            lockCount(mtxCountFinishedTask);
                        FinishedTask++;
                    }
                }
            });}
    }
    ~ThreadPool(){
        {
            std::unique_lock<std::mutex> lock(mtx);
            stop = true;
        }
        condition.notify_all();
        for(auto& t : threads){
            if(t.joinable())t.join();
        }
    }
    template<class F, class... Args>
    void AddNormalTask(F &&f, Args&&... args){
        std::function<void()>task=\
            std::bind(std::forward<F>(f),std::forward<Args>(args)...);
        //注意控制队列中的任务数量，可能线程的处理速度跟不上任务的增加速度，导致任务队列膨胀
        {
            std::unique_lock<std::mutex> lockAdd(mtxAddTask);
            conditionAdd.wait(lockAdd,[this] {
                return tasks.size()<maxTask;
            });
        }
        {
            //这里加锁是为了保证任务队列的安全，防止其他线程在操作任务队列时出错；
            //只有一个线程在操作任务队列，其他线程只能等待；
            std::unique_lock<std::mutex>lockAdd(mtxAddTask);
            tasks.emplace(std::move(task));
            AddTask++;
            condition.notify_one();
        }
    }
};
#endif

